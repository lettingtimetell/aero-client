//! The allocation-only bidi/font path used by the vanilla font renderer.
//!
//! Minecraft 1.8.9 calls ICU with `ArabicShaping(8)`, then constructs a `Bidi` with
//! `LEVEL_DEFAULT_RTL` (`127`), selects `REORDER_DEFAULT` (`0`), and writes with
//! `DO_MIRRORING` (`2`).  The game cannot assume that ICU is available on every client target, so
//! this module keeps the small, self-contained Arabic shaping and UAX #9 implementation beside
//! the renderer.  It intentionally documents the remaining data-table limitation below instead
//! of pretending that a hand-maintained Unicode table is a complete ICU replacement.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BidiClass {
    LeftToRight,
    RightToLeft,
    ArabicLetter,
    EuropeanNumber,
    ArabicNumber,
    EuropeanSeparator,
    CommonSeparator,
    EuropeanTerminator,
    ParagraphSeparator,
    SegmentSeparator,
    WhiteSpace,
    OtherNeutral,
    LeftToRightEmbedding,
    LeftToRightOverride,
    RightToLeftEmbedding,
    RightToLeftOverride,
    PopDirectionalFormat,
    NonspacingMark,
    BoundaryNeutral,
    LeftToRightIsolate,
    RightToLeftIsolate,
    FirstStrongIsolate,
    PopDirectionalIsolate,
}

impl BidiClass {
    fn is_neutral(self) -> bool {
        matches!(
            self,
            Self::ParagraphSeparator
                | Self::SegmentSeparator
                | Self::WhiteSpace
                | Self::OtherNeutral
                | Self::LeftToRightIsolate
                | Self::RightToLeftIsolate
                | Self::FirstStrongIsolate
                | Self::PopDirectionalIsolate
        )
    }
}

#[derive(Debug, Clone)]
struct Item {
    ch: char,
    original: BidiClass,
    class: BidiClass,
    level: u8,
}

/// Reorder and shape a string as the Java renderer's ICU path does for ordinary client text.
#[must_use]
pub fn reorder(text: &str) -> String {
    let shaped = shape_arabic(text);
    let chars: Vec<char> = shaped.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    // FontRenderer receives one line in almost every call, but Bidi still treats paragraph
    // separators as boundaries. Keep each separator with the paragraph before it.
    let mut output = String::with_capacity(shaped.len());
    let mut start = 0;
    for (index, ch) in chars.iter().copied().enumerate() {
        if matches!(bidi_class(ch), BidiClass::ParagraphSeparator) {
            output.push_str(&reorder_paragraph(&chars[start..=index]));
            start = index + 1;
        }
    }
    if start < chars.len() {
        output.push_str(&reorder_paragraph(&chars[start..]));
    }
    output
}

/// Reorder UTF-16 units, matching the renderer's Java-compatible public walkers.
pub(crate) fn reorder_units(units: &[u16]) -> Vec<u16> {
    let text = String::from_utf16_lossy(units);
    reorder(&text).encode_utf16().collect()
}

fn reorder_paragraph(chars: &[char]) -> String {
    let mut items: Vec<Item> = chars
        .iter()
        .copied()
        .map(|ch| {
            let class = bidi_class(ch);
            Item {
                ch,
                original: class,
                class,
                level: 0,
            }
        })
        .collect();
    if items.is_empty() {
        return String::new();
    }

    let base_level = paragraph_level(&items);
    resolve_first_strong_isolates(&mut items);
    resolve_explicit_levels(&mut items, base_level);
    resolve_weak_types(&mut items, base_level);
    resolve_brackets_and_neutrals(&mut items, base_level);
    resolve_implicit_levels(&mut items);
    align_directional_control_levels(&mut items);
    align_boundary_neutral_levels(&mut items, base_level);
    reset_line_end_levels(&mut items, base_level);

    let visual = visual_order(&items, base_level);
    let mut output = String::with_capacity(chars.len());
    for index in visual {
        let item = &items[index];
        output.push(if item.level & 1 != 0 {
            mirrored(item.ch)
        } else {
            item.ch
        });
    }
    output
}

fn paragraph_level(items: &[Item]) -> u8 {
    let mut isolate_depth = 0usize;
    for item in items {
        match item.original {
            BidiClass::LeftToRightIsolate
            | BidiClass::RightToLeftIsolate
            | BidiClass::FirstStrongIsolate => isolate_depth += 1,
            BidiClass::PopDirectionalIsolate => isolate_depth = isolate_depth.saturating_sub(1),
            BidiClass::LeftToRight if isolate_depth == 0 => return 0,
            BidiClass::RightToLeft | BidiClass::ArabicLetter if isolate_depth == 0 => return 1,
            _ => {}
        }
    }
    // Java passes LEVEL_DEFAULT_RTL (127), so an entirely neutral paragraph is RTL.
    1
}

fn resolve_first_strong_isolates(items: &mut [Item]) {
    for index in 0..items.len() {
        if items[index].original != BidiClass::FirstStrongIsolate {
            continue;
        }
        let mut depth = 0usize;
        let mut found = None;
        for candidate in items.iter().skip(index + 1) {
            match candidate.original {
                BidiClass::LeftToRightIsolate
                | BidiClass::RightToLeftIsolate
                | BidiClass::FirstStrongIsolate => depth += 1,
                BidiClass::PopDirectionalIsolate if depth > 0 => depth -= 1,
                BidiClass::PopDirectionalIsolate => break,
                BidiClass::LeftToRight if depth == 0 => {
                    found = Some(BidiClass::LeftToRightIsolate);
                    break;
                }
                BidiClass::RightToLeft | BidiClass::ArabicLetter if depth == 0 => {
                    found = Some(BidiClass::RightToLeftIsolate);
                    break;
                }
                _ => {}
            }
        }
        if let Some(class) = found {
            items[index].class = class;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingStatus {
    level: u8,
    override_class: Option<BidiClass>,
    isolate: bool,
}

fn next_explicit_level(level: u8, rtl: bool) -> Option<u8> {
    let parity = if rtl { 1 } else { 0 };
    let candidate = if level & 1 == parity {
        level.saturating_add(2)
    } else {
        level.saturating_add(1)
    };
    (candidate <= 61).then_some(candidate)
}

fn resolve_explicit_levels(items: &mut [Item], base_level: u8) {
    let mut stack = vec![EmbeddingStatus {
        level: base_level,
        override_class: None,
        isolate: false,
    }];
    let mut overflow_isolate_count = 0u8;
    let mut overflow_embedding_count = 0u8;
    let mut valid_isolate_count = 0u8;

    for item in items.iter_mut() {
        let current = stack.last().copied().unwrap_or(EmbeddingStatus {
            level: base_level,
            override_class: None,
            isolate: false,
        });
        match item.original {
            BidiClass::RightToLeftEmbedding
            | BidiClass::LeftToRightEmbedding
            | BidiClass::RightToLeftOverride
            | BidiClass::LeftToRightOverride
            | BidiClass::RightToLeftIsolate
            | BidiClass::LeftToRightIsolate
            | BidiClass::FirstStrongIsolate => {
                let isolate = matches!(
                    item.original,
                    BidiClass::RightToLeftIsolate
                        | BidiClass::LeftToRightIsolate
                        | BidiClass::FirstStrongIsolate
                );
                let rtl = matches!(
                    item.class,
                    BidiClass::RightToLeftIsolate
                        | BidiClass::RightToLeftEmbedding
                        | BidiClass::RightToLeftOverride
                );
                let new_level = next_explicit_level(current.level, rtl);
                item.level = current.level;
                if isolate {
                    if let Some(new_level) = new_level {
                        if overflow_isolate_count == 0 && overflow_embedding_count == 0 {
                            stack.push(EmbeddingStatus {
                                level: new_level,
                                override_class: None,
                                isolate: true,
                            });
                            valid_isolate_count = valid_isolate_count.saturating_add(1);
                        } else {
                            overflow_isolate_count = overflow_isolate_count.saturating_add(1);
                        }
                    } else {
                        overflow_isolate_count = overflow_isolate_count.saturating_add(1);
                    }
                } else if let Some(new_level) = new_level {
                    if overflow_isolate_count == 0 && overflow_embedding_count == 0 {
                        item.level = new_level;
                        stack.push(EmbeddingStatus {
                            level: new_level,
                            override_class: match item.original {
                                BidiClass::RightToLeftOverride => Some(BidiClass::RightToLeft),
                                BidiClass::LeftToRightOverride => Some(BidiClass::LeftToRight),
                                _ => None,
                            },
                            isolate: false,
                        });
                    } else if overflow_isolate_count == 0 {
                        overflow_embedding_count = overflow_embedding_count.saturating_add(1);
                    }
                } else if overflow_isolate_count == 0 {
                    overflow_embedding_count = overflow_embedding_count.saturating_add(1);
                }
                // X9 removes these classes from type resolution, but ICU retains their UTF-16
                // code units when writeReordered is called without REMOVE_BIDI_CONTROLS.
                item.class = BidiClass::BoundaryNeutral;
            }
            BidiClass::PopDirectionalIsolate => {
                if overflow_isolate_count > 0 {
                    overflow_isolate_count -= 1;
                } else if valid_isolate_count > 0 {
                    overflow_embedding_count = 0;
                    while stack.len() > 1 {
                        let popped = stack.pop().unwrap();
                        if popped.isolate {
                            break;
                        }
                    }
                    valid_isolate_count -= 1;
                }
                let current = stack.last().copied().unwrap_or(EmbeddingStatus {
                    level: base_level,
                    override_class: None,
                    isolate: false,
                });
                item.level = current.level;
                item.class = current.override_class.unwrap_or(BidiClass::OtherNeutral);
            }
            BidiClass::PopDirectionalFormat => {
                if overflow_isolate_count == 0 {
                    if overflow_embedding_count > 0 {
                        overflow_embedding_count -= 1;
                    } else if stack.len() > 1 && !current.isolate {
                        stack.pop();
                    }
                }
                item.level = current.level;
                item.class = BidiClass::BoundaryNeutral;
            }
            BidiClass::BoundaryNeutral => item.level = current.level,
            _ => {
                item.level = current.level;
                if let Some(override_class) = current.override_class {
                    item.class = override_class;
                }
            }
        }
    }
}

fn resolve_weak_types(items: &mut [Item], base_level: u8) {
    // W1: NSM inherits the previous type. An isolate boundary has ON as its synthetic predecessor.
    let mut previous = if base_level & 1 == 0 {
        BidiClass::LeftToRight
    } else {
        BidiClass::RightToLeft
    };
    for item in items.iter_mut() {
        if item.class == BidiClass::BoundaryNeutral {
            continue;
        }
        if item.class == BidiClass::NonspacingMark {
            item.class = match previous {
                BidiClass::LeftToRightIsolate
                | BidiClass::RightToLeftIsolate
                | BidiClass::FirstStrongIsolate
                | BidiClass::PopDirectionalIsolate => BidiClass::OtherNeutral,
                previous => previous,
            };
        }
        previous = item.class;
    }

    // W2/W3: Arabic letters turn into R, and EN following the last AL becomes AN.
    let mut last_strong = if base_level & 1 == 0 {
        BidiClass::LeftToRight
    } else {
        BidiClass::RightToLeft
    };
    for item in items.iter_mut() {
        if item.class == BidiClass::BoundaryNeutral {
            continue;
        }
        if item.class == BidiClass::EuropeanNumber && last_strong == BidiClass::ArabicLetter {
            item.class = BidiClass::ArabicNumber;
        }
        if item.class == BidiClass::ArabicLetter {
            item.class = BidiClass::RightToLeft;
        }
        match item.class {
            BidiClass::LeftToRight | BidiClass::RightToLeft => last_strong = item.class,
            _ => {}
        }
    }

    // W4: separators between matching number types become that number type.
    for index in 1..items.len().saturating_sub(1) {
        if items[index].class == BidiClass::EuropeanSeparator
            && items[index - 1].class == BidiClass::EuropeanNumber
            && items[index + 1].class == BidiClass::EuropeanNumber
        {
            items[index].class = BidiClass::EuropeanNumber;
        } else if items[index].class == BidiClass::CommonSeparator {
            if items[index - 1].class == BidiClass::EuropeanNumber
                && items[index + 1].class == BidiClass::EuropeanNumber
            {
                items[index].class = BidiClass::EuropeanNumber;
            } else if items[index - 1].class == BidiClass::ArabicNumber
                && items[index + 1].class == BidiClass::ArabicNumber
            {
                items[index].class = BidiClass::ArabicNumber;
            }
        }
    }

    // W5: ET sequences adjacent to EN become EN.
    let mut index = 0;
    while index < items.len() {
        if items[index].class != BidiClass::EuropeanTerminator {
            index += 1;
            continue;
        }
        let start = index;
        while index < items.len() && items[index].class == BidiClass::EuropeanTerminator {
            index += 1;
        }
        if (start > 0 && items[start - 1].class == BidiClass::EuropeanNumber)
            || (index < items.len() && items[index].class == BidiClass::EuropeanNumber)
        {
            for item in &mut items[start..index] {
                item.class = BidiClass::EuropeanNumber;
            }
        }
    }

    // W6: remaining ES/ET/CS become ON.
    for item in items.iter_mut() {
        if matches!(
            item.class,
            BidiClass::EuropeanSeparator
                | BidiClass::EuropeanTerminator
                | BidiClass::CommonSeparator
        ) {
            item.class = BidiClass::OtherNeutral;
        }
    }

    // W7: EN following an L strong type becomes L.
    let mut last_ltr = base_level & 1 == 0;
    for item in items.iter_mut() {
        match item.class {
            BidiClass::EuropeanNumber if last_ltr => item.class = BidiClass::LeftToRight,
            BidiClass::LeftToRight => last_ltr = true,
            BidiClass::RightToLeft => last_ltr = false,
            _ => {}
        }
    }
}

fn resolve_brackets_and_neutrals(items: &mut [Item], base_level: u8) {
    // N0 paired bracket resolution. The table covers the ASCII and common Unicode brackets used
    // by Minecraft labels; all other ON characters still receive the ordinary N1/N2 treatment.
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut pairs = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if item.class != BidiClass::OtherNeutral {
            continue;
        }
        if is_open_bracket(item.ch) {
            if stack.len() < 63 {
                stack.push((item.ch, index));
            }
        } else if let Some(opening) = matching_opening(item.ch) {
            if let Some(position) = stack
                .iter()
                .rposition(|(candidate, _)| *candidate == opening)
            {
                let (_, start) = stack[position];
                stack.truncate(position);
                pairs.push((start, index));
            }
        }
    }

    for (start, end) in pairs {
        let embedding = if base_level & 1 == 0 {
            BidiClass::LeftToRight
        } else {
            BidiClass::RightToLeft
        };
        let opposite = if embedding == BidiClass::LeftToRight {
            BidiClass::RightToLeft
        } else {
            BidiClass::LeftToRight
        };
        let mut found_embedding = false;
        let mut found_opposite = false;
        for item in &items[start + 1..end] {
            match item.class {
                class if class == embedding => found_embedding = true,
                class if class == opposite => found_opposite = true,
                BidiClass::EuropeanNumber | BidiClass::ArabicNumber => {
                    if embedding == BidiClass::RightToLeft {
                        found_embedding = true;
                    } else {
                        found_opposite = true;
                    }
                }
                _ => {}
            }
            if found_embedding {
                break;
            }
        }
        if found_embedding {
            items[start].class = embedding;
            items[end].class = embedding;
        } else if found_opposite {
            // The preceding strong type establishes context when the bracket contents are only
            // opposite-direction text. This is the common `(אב)` / `אב (abc)` case.
            let previous = items[..start]
                .iter()
                .rev()
                .find_map(|item| match item.class {
                    BidiClass::LeftToRight | BidiClass::RightToLeft => Some(item.class),
                    BidiClass::EuropeanNumber | BidiClass::ArabicNumber => {
                        Some(BidiClass::RightToLeft)
                    }
                    _ => None,
                })
                .unwrap_or(embedding);
            items[start].class = previous;
            items[end].class = previous;
        }
    }

    // N1/N2: resolve each neutral span from its surrounding strong types or the embedding level.
    let embedding = if base_level & 1 == 0 {
        BidiClass::LeftToRight
    } else {
        BidiClass::RightToLeft
    };
    let mut index = 0;
    while index < items.len() {
        if !items[index].class.is_neutral() && items[index].class != BidiClass::BoundaryNeutral {
            index += 1;
            continue;
        }
        let start = index;
        while index < items.len()
            && (items[index].class.is_neutral() || items[index].class == BidiClass::BoundaryNeutral)
        {
            index += 1;
        }
        let previous = items[..start]
            .iter()
            .rev()
            .find_map(strong_class_for_neutral)
            .unwrap_or(embedding);
        let next = items[index..]
            .iter()
            .find_map(strong_class_for_neutral)
            .unwrap_or(embedding);
        let resolved = if previous == next {
            previous
        } else {
            embedding
        };
        for item in &mut items[start..index] {
            if item.class != BidiClass::BoundaryNeutral {
                item.class = resolved;
            }
        }
    }
}

fn strong_class_for_neutral(item: &Item) -> Option<BidiClass> {
    match item.class {
        BidiClass::LeftToRight | BidiClass::RightToLeft => Some(item.class),
        BidiClass::EuropeanNumber | BidiClass::ArabicNumber => Some(BidiClass::RightToLeft),
        _ => None,
    }
}

fn resolve_implicit_levels(items: &mut [Item]) {
    for item in items.iter_mut() {
        match (item.level & 1 != 0, item.class) {
            (false, BidiClass::RightToLeft)
            | (true, BidiClass::LeftToRight)
            | (true, BidiClass::EuropeanNumber)
            | (true, BidiClass::ArabicNumber) => item.level = item.level.saturating_add(1),
            (false, BidiClass::EuropeanNumber) | (false, BidiClass::ArabicNumber) => {
                item.level = item.level.saturating_add(2)
            }
            _ => {}
        }
    }
}

fn align_directional_control_levels(items: &mut [Item]) {
    let mut embedding_stack = Vec::new();
    let mut isolate_stack = Vec::new();
    let mut pairs = Vec::new();
    for (index, item) in items.iter().enumerate() {
        match item.original {
            BidiClass::RightToLeftEmbedding
            | BidiClass::LeftToRightEmbedding
            | BidiClass::RightToLeftOverride
            | BidiClass::LeftToRightOverride => embedding_stack.push(index),
            BidiClass::RightToLeftIsolate
            | BidiClass::LeftToRightIsolate
            | BidiClass::FirstStrongIsolate => isolate_stack.push(index),
            BidiClass::PopDirectionalFormat => {
                if let Some(start) = embedding_stack.pop() {
                    pairs.push((start, index));
                }
            }
            BidiClass::PopDirectionalIsolate => {
                if let Some(start) = isolate_stack.pop() {
                    pairs.push((start, index));
                }
            }
            _ => {}
        }
    }
    let mut paired = std::collections::HashSet::new();
    for (start, end) in pairs {
        paired.insert(start);
        paired.insert(end);
        if let Some(level) = items[start + 1..end].iter().map(|item| item.level).max() {
            items[start].level = level;
            items[end].level = level;
        }
    }

    // ICU also retains unmatched explicit/isolate controls. With no closing/opening partner,
    // their level follows the next real item (or the previous item when trailing), so a lone
    // control still travels with the directional run it affects.
    for index in 0..items.len() {
        if paired.contains(&index) || !is_directional_control(items[index].original) {
            continue;
        }
        let next_level = items[index + 1..]
            .iter()
            .find(|item| !is_directional_control(item.original))
            .map(|item| item.level);
        let previous_level = items[..index]
            .iter()
            .rev()
            .find(|item| !is_directional_control(item.original))
            .map(|item| item.level);
        if let Some(level) = next_level.or(previous_level) {
            items[index].level = level;
        }
    }
}

fn is_directional_control(class: BidiClass) -> bool {
    matches!(
        class,
        BidiClass::LeftToRightEmbedding
            | BidiClass::LeftToRightOverride
            | BidiClass::RightToLeftEmbedding
            | BidiClass::RightToLeftOverride
            | BidiClass::PopDirectionalFormat
            | BidiClass::LeftToRightIsolate
            | BidiClass::RightToLeftIsolate
            | BidiClass::FirstStrongIsolate
            | BidiClass::PopDirectionalIsolate
    )
}

/// ICU retains X9 boundary-neutral code units in the visual output. Their level follows the next
/// non-boundary item (or the preceding item when they trail a line), which makes format controls
/// travel with the run they precede without letting a later RTL run pull an intervening LTR BN
/// across the line.
fn align_boundary_neutral_levels(items: &mut [Item], base_level: u8) {
    for index in 0..items.len() {
        if items[index].original != BidiClass::BoundaryNeutral {
            continue;
        }
        let next_level = items[index + 1..]
            .iter()
            .find(|item| item.original != BidiClass::BoundaryNeutral)
            .map(|item| item.level);
        let previous_level = items[..index]
            .iter()
            .rev()
            .find(|item| item.original != BidiClass::BoundaryNeutral)
            .map(|item| item.level);
        items[index].level = next_level.or(previous_level).unwrap_or(base_level);
    }
}

fn reset_line_end_levels(items: &mut [Item], base_level: u8) {
    // L1: reset trailing whitespace and segment separators to the paragraph level. Do not reset a
    // paragraph separator itself here; it is already retained by the paragraph splitter.
    let mut end = items.len();
    while end > 0
        && matches!(
            items[end - 1].original,
            BidiClass::WhiteSpace | BidiClass::SegmentSeparator | BidiClass::BoundaryNeutral
        )
    {
        end -= 1;
    }
    for item in &mut items[end..] {
        item.level = base_level;
    }
}

fn visual_order(items: &[Item], base_level: u8) -> Vec<usize> {
    let order = visual_order_indices(items);
    relocate_directional_controls(order, items, base_level)
}

fn visual_order_indices(items: &[Item]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    let Some(max_level) = items.iter().map(|item| item.level).max() else {
        return order;
    };
    let min_odd = items
        .iter()
        .map(|item| item.level)
        .filter(|level| level & 1 != 0)
        .min();
    let Some(min_odd) = min_odd else {
        return order;
    };

    for level in (min_odd..=max_level).rev() {
        let mut index = 0;
        while index < order.len() {
            if items[order[index]].level < level {
                index += 1;
                continue;
            }
            let start = index;
            while index < order.len() && items[order[index]].level >= level {
                index += 1;
            }
            order[start..index].reverse();
        }
    }
    order
}

fn relocate_directional_controls(order: Vec<usize>, items: &[Item], base_level: u8) -> Vec<usize> {
    let mut pairs = Vec::new();
    let mut embedding_stack = Vec::new();
    let mut isolate_stack = Vec::new();
    for (index, item) in items.iter().enumerate() {
        match item.original {
            BidiClass::RightToLeftEmbedding
            | BidiClass::LeftToRightEmbedding
            | BidiClass::RightToLeftOverride
            | BidiClass::LeftToRightOverride => embedding_stack.push(index),
            BidiClass::RightToLeftIsolate
            | BidiClass::LeftToRightIsolate
            | BidiClass::FirstStrongIsolate => isolate_stack.push(index),
            BidiClass::PopDirectionalFormat => {
                if let Some(start) = embedding_stack.pop() {
                    pairs.push((start, index));
                }
            }
            BidiClass::PopDirectionalIsolate => {
                if let Some(start) = isolate_stack.pop() {
                    pairs.push((start, index));
                }
            }
            _ => {}
        }
    }
    if pairs.is_empty() {
        return order;
    }

    let control_indices: std::collections::HashSet<usize> = pairs
        .iter()
        .flat_map(|(start, end)| [*start, *end])
        .collect();
    let noncontrols: Vec<usize> = order
        .iter()
        .copied()
        .filter(|index| !control_indices.contains(index))
        .collect();
    let mut insertions: Vec<Vec<usize>> = vec![Vec::new(); noncontrols.len() + 1];
    for (start, end) in pairs {
        let content: Vec<usize> = noncontrols
            .iter()
            .copied()
            .filter(|index| *index > start && *index < end)
            .collect();
        if content.is_empty() {
            continue;
        }
        let first = noncontrols
            .iter()
            .position(|index| *index == content[0])
            .unwrap_or(0);
        let last = noncontrols
            .iter()
            .position(|index| *index == *content.last().unwrap())
            .unwrap_or(first);
        let content_is_rtl = content.iter().any(|index| items[*index].level & 1 != 0);
        if content_is_rtl {
            if base_level & 1 == 0 {
                insertions[last + 1].extend([start, end]);
            } else {
                insertions[first].push(end);
                insertions[last + 1].push(start);
            }
        } else {
            insertions[first].push(start);
            insertions[last + 1].push(end);
        }
    }
    let mut relocated = Vec::with_capacity(order.len());
    for (position, index) in noncontrols.into_iter().enumerate() {
        relocated.extend(insertions[position].iter().copied());
        relocated.push(index);
    }
    relocated.extend(insertions.last().into_iter().flatten().copied());
    relocated
}

fn bidi_class(ch: char) -> BidiClass {
    let c = ch as u32;
    match c {
        0x000a | 0x000d | 0x001c..=0x001e | 0x0085 => BidiClass::ParagraphSeparator,
        0x0009 | 0x000b | 0x001f => BidiClass::SegmentSeparator,
        0x000c => BidiClass::WhiteSpace,
        0x202a => BidiClass::LeftToRightEmbedding,
        0x202b => BidiClass::RightToLeftEmbedding,
        0x202c => BidiClass::PopDirectionalFormat,
        0x202d => BidiClass::LeftToRightOverride,
        0x202e => BidiClass::RightToLeftOverride,
        0x2066 => BidiClass::LeftToRightIsolate,
        0x2067 => BidiClass::RightToLeftIsolate,
        0x2068 => BidiClass::FirstStrongIsolate,
        0x2069 => BidiClass::PopDirectionalIsolate,
        0x200e => BidiClass::LeftToRight,
        0x200f => BidiClass::RightToLeft,
        0x0000..=0x0008
        | 0x000e..=0x001b
        | 0x007f..=0x0084
        | 0x0086..=0x009f
        | 0x00ad
        | 0x200b..=0x200d
        | 0x2060..=0x2065
        | 0x206a..=0x206f
        | 0xfeff
        | 0xfff0..=0xfff8
        | 0xfffe..=0xffff => BidiClass::BoundaryNeutral,
        0x0030..=0x0039 | 0x06f0..=0x06f9 | 0xff10..=0xff19 => BidiClass::EuropeanNumber,
        0x0600..=0x0604 | 0x0660..=0x0669 | 0x066b..=0x066c | 0x06dd => BidiClass::ArabicNumber,
        0x066a => BidiClass::EuropeanTerminator,
        0x002b | 0x002d | 0x2212 => BidiClass::EuropeanSeparator,
        0x002c | 0x002e | 0x002f | 0x003a => BidiClass::CommonSeparator,
        0x0023 | 0x0024 | 0x0025 | 0x00a2..=0x00a5 => BidiClass::EuropeanTerminator,
        0x0591..=0x05bd
        | 0x05bf
        | 0x05c1..=0x05c2
        | 0x05c4..=0x05c5
        | 0x0610..=0x061a
        | 0x064b..=0x065f
        | 0x0670
        | 0x06d6..=0x06e4
        | 0x06e7..=0x06ed
        | 0x0711
        | 0x0730..=0x074a
        | 0x07a6..=0x07b0
        | 0x07eb..=0x07f3
        | 0x0816..=0x0819
        | 0x081b..=0x0823
        | 0x0825..=0x0827
        | 0x0829..=0x082d
        | 0x0859..=0x085b
        | 0x08e4..=0x08fe
        | 0x0900..=0x0903
        | 0x093a..=0x094f
        | 0x0951..=0x0957
        | 0x0962..=0x0963
        | 0x0981..=0x0983
        | 0x09bc
        | 0x09be..=0x09cd
        | 0x09d7
        | 0x09e2..=0x09e3
        | 0x0300..=0x036f
        | 0x0483..=0x0489
        | 0x1ab0..=0x1aff
        | 0x1dc0..=0x1dff
        | 0x20d0..=0x20ff
        | 0xfe20..=0xfe2f => BidiClass::NonspacingMark,
        0xfb1e | 0xfe00..=0xfe0f => BidiClass::NonspacingMark,
        0x07f6..=0x07f9 => BidiClass::OtherNeutral,
        0xfb29 | 0xfe62..=0xfe63 => BidiClass::EuropeanSeparator,
        0xfd3e..=0xfd3f | 0xfdfd | 0xfe10..=0xfe19 | 0xfe30..=0xfe4f => BidiClass::OtherNeutral,
        0xfdd0..=0xfdef => BidiClass::BoundaryNeutral,
        0x0590 | 0x05be | 0x05c0 | 0x05c3 | 0x05c6 | 0x05c8..=0x05ff | 0xfb1d..=0xfb4f => {
            BidiClass::RightToLeft
        }
        0x0605
        | 0x0608
        | 0x060b
        | 0x060d
        | 0x061b..=0x064a
        | 0x066d..=0x066f
        | 0x0700..=0x0710
        | 0x0712..=0x072f
        | 0x074b..=0x07a5
        | 0x0671..=0x06d5
        | 0x06e5..=0x06e6
        | 0x06ee..=0x06ef
        | 0x06fa..=0x06ff
        | 0x0780..=0x07bf
        | 0x07c0..=0x07ff
        | 0x0800..=0x083f
        | 0x0840..=0x085f
        | 0x0860..=0x086f
        | 0x0870..=0x089f
        | 0x08a0..=0x08ff
        | 0xfb50..=0xfdff
        | 0xfe70..=0xfe7f
        | 0xfe80..=0xfefc
        | 0xfefd..=0xfefe
        | 0x1ee00..=0x1eeef
        | 0x1eef2..=0x1eeff => BidiClass::ArabicLetter,
        _ if supplementary_boundary_neutral(c) => BidiClass::BoundaryNeutral,
        _ if supplementary_arabic_number(c) => BidiClass::ArabicNumber,
        _ if supplementary_european_number(c) => BidiClass::EuropeanNumber,
        _ if supplementary_nonspacing_mark(c) => BidiClass::NonspacingMark,
        _ if supplementary_other_neutral(c) => BidiClass::OtherNeutral,
        _ if supplementary_right_to_left(c) => BidiClass::RightToLeft,
        _ if ch.is_whitespace() => BidiClass::WhiteSpace,
        _ if is_other_neutral(ch) => BidiClass::OtherNeutral,
        _ => BidiClass::LeftToRight,
    }
}

// ICU 51.2's bidi property data assigns directional types to a handful of supplementary scripts
// and musical/emoji blocks. Keep these ranges explicit instead of treating every non-BMP scalar
// as LTR; this is the same data boundary as the 1.8.9 bundled ICU, not a modern-ICU expansion.
fn supplementary_boundary_neutral(c: u32) -> bool {
    matches!(
        c,
        0x01d173..=0x01d17a
            | 0x0e0000..=0x0e00ff
            | 0x0e01f0..=0x0e0fff
            | 0x01fffe..=0x01ffff
            | 0x02fffe..=0x02ffff
            | 0x03fffe..=0x03ffff
            | 0x04fffe..=0x04ffff
            | 0x05fffe..=0x05ffff
            | 0x06fffe..=0x06ffff
            | 0x07fffe..=0x07ffff
            | 0x08fffe..=0x08ffff
            | 0x09fffe..=0x09ffff
            | 0x0afffe..=0x0affff
            | 0x0bfffe..=0x0bffff
            | 0x0cfffe..=0x0cffff
            | 0x0dfffe..=0x0dffff
            | 0x0efffe..=0x0effff
            | 0x0ffffe..=0x0fffff
            | 0x10fffe..=0x10ffff
    )
}

fn supplementary_arabic_number(c: u32) -> bool {
    (0x010e60..=0x010e7e).contains(&c)
}

fn supplementary_european_number(c: u32) -> bool {
    matches!(c, 0x01d7ce..=0x01d7ff | 0x01f100..=0x01f10a)
}

fn supplementary_nonspacing_mark(c: u32) -> bool {
    matches!(
        c,
        0x010a01..=0x010a03
            | 0x010a05..=0x010a06
            | 0x010a0c..=0x010a0f
            | 0x010a38..=0x010a3a
            | 0x010a3f
            | 0x011001
            | 0x011038..=0x011046
            | 0x011052..=0x011065
            | 0x011080..=0x011081
            | 0x0110b3..=0x0110b6
            | 0x0110b9..=0x0110ba
            | 0x011100..=0x011102
            | 0x011127..=0x01112b
            | 0x01112d..=0x011134
            | 0x011180..=0x011181
            | 0x0111b6..=0x0111be
            | 0x0116ab
            | 0x0116ad
            | 0x0116b0..=0x0116b5
            | 0x0116b7
            | 0x016f8f..=0x016f92
            | 0x01d167..=0x01d169
            | 0x01d17b..=0x01d182
            | 0x01d185..=0x01d18b
            | 0x01d1aa..=0x01d1ad
            | 0x01d242..=0x01d244
            | 0x0e0100..=0x0e01ef
    )
}

fn supplementary_other_neutral(c: u32) -> bool {
    matches!(
        c,
        0x010101
            | 0x010140..=0x01018a
            | 0x010190..=0x01019b
            | 0x01091f
            | 0x010b39..=0x010b3f
            | 0x01d200..=0x01d241
            | 0x01d245
            | 0x01d300..=0x01d356
            | 0x01d6db
            | 0x01d715
            | 0x01d74f
            | 0x01d789
            | 0x01d7c3
            | 0x01eef0..=0x01eef1
            | 0x01f000..=0x01f02b
            | 0x01f030..=0x01f093
            | 0x01f0a0..=0x01f0ae
            | 0x01f0b1..=0x01f0be
            | 0x01f0c1..=0x01f0cf
            | 0x01f0d1..=0x01f0df
            | 0x01f16a..=0x01f16b
            | 0x01f300..=0x01f320
            | 0x01f330..=0x01f335
            | 0x01f337..=0x01f37c
            | 0x01f380..=0x01f393
            | 0x01f3a0..=0x01f3c4
            | 0x01f3c6..=0x01f3ca
            | 0x01f3e0..=0x01f3f0
            | 0x01f400..=0x01f43e
            | 0x01f440
            | 0x01f442..=0x01f4f7
            | 0x01f4f9..=0x01f4fc
            | 0x01f500..=0x01f53d
            | 0x01f540..=0x01f543
            | 0x01f550..=0x01f567
            | 0x01f5fb..=0x01f640
            | 0x01f645..=0x01f64f
            | 0x01f680..=0x01f6c5
            | 0x01f700..=0x01f773
    )
}

fn supplementary_right_to_left(c: u32) -> bool {
    matches!(
        c,
        0x010800..=0x010e5f
            | 0x010e7f..=0x010fff
            | 0x01e800..=0x01edff
            | 0x01ef00..=0x01efff
    )
}

fn is_other_neutral(ch: char) -> bool {
    let c = ch as u32;
    matches!(
        c,
        0x0000..=0x0008
            | 0x000b..=0x000c
            | 0x000e..=0x001b
            | 0x0021..=0x0022
            | 0x0026..=0x0029
            | 0x003b..=0x0040
            | 0x005b..=0x0060
            | 0x007b..=0x007f
            | 0x00a1
            | 0x00a6..=0x00bf
            | 0x02b0..=0x02ff
            | 0x0370..=0x037f
            | 0x055a..=0x055f
            | 0x060c
            | 0x061b
            | 0x061f
            | 0x066a..=0x066d
            | 0x06d4
            | 0x0700..=0x070a
            | 0x07f4..=0x07f5
            | 0x0830..=0x083e
            | 0x085e
            | 0x2000..=0x206f
            | 0x2e00..=0x2e7f
            | 0x3000..=0x303f
            | 0xff01..=0xff65
    )
}

fn is_open_bracket(ch: char) -> bool {
    matches!(
        ch,
        '(' | '['
            | '{'
            | '<'
            | '\u{00ab}'
            | '\u{2039}'
            | '\u{2045}'
            | '\u{207d}'
            | '\u{208d}'
            | '\u{2208}'
            | '\u{220b}'
            | '\u{2329}'
            | '\u{27e8}'
            | '\u{3008}'
            | '\u{300a}'
            | '\u{300c}'
            | '\u{300e}'
            | '\u{3010}'
            | '\u{3014}'
            | '\u{3016}'
            | '\u{3018}'
            | '\u{301a}'
    )
}

fn matching_opening(ch: char) -> Option<char> {
    Some(match ch {
        ')' => '(',
        ']' => '[',
        '}' => '{',
        '>' => '<',
        '\u{00bb}' => '\u{00ab}',
        '\u{203a}' => '\u{2039}',
        '\u{2046}' => '\u{2045}',
        '\u{207e}' => '\u{207d}',
        '\u{208e}' => '\u{208d}',
        '\u{232a}' => '\u{2329}',
        '\u{27e9}' => '\u{27e8}',
        '\u{3009}' => '\u{3008}',
        '\u{300b}' => '\u{300a}',
        '\u{300d}' => '\u{300c}',
        '\u{300f}' => '\u{300e}',
        '\u{3011}' => '\u{3010}',
        '\u{3015}' => '\u{3014}',
        '\u{3017}' => '\u{3016}',
        '\u{3019}' => '\u{3018}',
        '\u{301b}' => '\u{301a}',
        _ => return None,
    })
}

fn mirrored(ch: char) -> char {
    match ch {
        '(' => ')',
        ')' => '(',
        '[' => ']',
        ']' => '[',
        '{' => '}',
        '}' => '{',
        '<' => '>',
        '>' => '<',
        '\u{00ab}' => '\u{00bb}',
        '\u{00bb}' => '\u{00ab}',
        '\u{2039}' => '\u{203a}',
        '\u{203a}' => '\u{2039}',
        '\u{2045}' => '\u{2046}',
        '\u{2046}' => '\u{2045}',
        '\u{207d}' => '\u{207e}',
        '\u{207e}' => '\u{207d}',
        '\u{208d}' => '\u{208e}',
        '\u{208e}' => '\u{208d}',
        '\u{2208}' => '\u{220b}',
        '\u{220b}' => '\u{2208}',
        '\u{2215}' => '\u{29f8}',
        '\u{221f}' => '\u{2bfe}',
        '\u{2229}' => '\u{222a}',
        '\u{222a}' => '\u{2229}',
        '\u{2234}' => '\u{2235}',
        '\u{2235}' => '\u{2234}',
        '\u{2243}' => '\u{22cd}',
        '\u{2252}' => '\u{2253}',
        '\u{2254}' => '\u{2255}',
        '\u{2264}' => '\u{2265}',
        '\u{2265}' => '\u{2264}',
        '\u{2266}' => '\u{2267}',
        '\u{2267}' => '\u{2266}',
        '\u{2268}' => '\u{2269}',
        '\u{2269}' => '\u{2268}',
        '\u{226a}' => '\u{226b}',
        '\u{226b}' => '\u{226a}',
        '\u{2282}' => '\u{2283}',
        '\u{2283}' => '\u{2282}',
        '\u{2286}' => '\u{2287}',
        '\u{2287}' => '\u{2286}',
        '\u{2298}' => '\u{29b8}',
        '\u{22a2}' => '\u{22a3}',
        '\u{22a6}' => '\u{2ade}',
        '\u{22a7}' => '\u{2adf}',
        '\u{22c2}' => '\u{22c3}',
        '\u{22c3}' => '\u{22c2}',
        '\u{22c4}' => '\u{22c5}',
        '\u{22c5}' => '\u{22c4}',
        '\u{2308}' => '\u{2309}',
        '\u{2309}' => '\u{2308}',
        '\u{230a}' => '\u{230b}',
        '\u{230b}' => '\u{230a}',
        '\u{2329}' => '\u{232a}',
        '\u{232a}' => '\u{2329}',
        '\u{27e8}' => '\u{27e9}',
        '\u{27e9}' => '\u{27e8}',
        '\u{27ea}' => '\u{27eb}',
        '\u{27eb}' => '\u{27ea}',
        '\u{27ec}' => '\u{27ed}',
        '\u{27ed}' => '\u{27ec}',
        '\u{27ee}' => '\u{27ef}',
        '\u{27ef}' => '\u{27ee}',
        '\u{2983}' => '\u{2984}',
        '\u{2984}' => '\u{2983}',
        '\u{2985}' => '\u{2986}',
        '\u{2986}' => '\u{2985}',
        '\u{2987}' => '\u{2988}',
        '\u{2988}' => '\u{2987}',
        '\u{2989}' => '\u{298a}',
        '\u{298a}' => '\u{2989}',
        '\u{298b}' => '\u{298c}',
        '\u{298c}' => '\u{298b}',
        '\u{298d}' => '\u{2990}',
        '\u{2990}' => '\u{298d}',
        '\u{298e}' => '\u{298f}',
        '\u{298f}' => '\u{298e}',
        '\u{2991}' => '\u{2992}',
        '\u{2992}' => '\u{2991}',
        '\u{2993}' => '\u{2994}',
        '\u{2994}' => '\u{2993}',
        '\u{2995}' => '\u{2996}',
        '\u{2996}' => '\u{2995}',
        '\u{2997}' => '\u{2998}',
        '\u{2998}' => '\u{2997}',
        '\u{29fc}' => '\u{29fd}',
        '\u{29fd}' => '\u{29fc}',
        '\u{3008}' => '\u{3009}',
        '\u{3009}' => '\u{3008}',
        '\u{300a}' => '\u{300b}',
        '\u{300b}' => '\u{300a}',
        '\u{300c}' => '\u{300d}',
        '\u{300d}' => '\u{300c}',
        '\u{300e}' => '\u{300f}',
        '\u{300f}' => '\u{300e}',
        '\u{3010}' => '\u{3011}',
        '\u{3011}' => '\u{3010}',
        '\u{3014}' => '\u{3015}',
        '\u{3015}' => '\u{3014}',
        '\u{3016}' => '\u{3017}',
        '\u{3017}' => '\u{3016}',
        '\u{3018}' => '\u{3019}',
        '\u{3019}' => '\u{3018}',
        '\u{301a}' => '\u{301b}',
        '\u{301b}' => '\u{301a}',
        _ => ch,
    }
}

#[derive(Debug, Clone, Copy)]
struct ArabicForm {
    isolated: char,
    final_form: char,
    initial: char,
    medial: char,
    joins_left: bool,
    joins_right: bool,
}

const fn form(
    isolated: char,
    final_form: char,
    initial: char,
    medial: char,
    joins_left: bool,
    joins_right: bool,
) -> ArabicForm {
    ArabicForm {
        isolated,
        final_form,
        initial,
        medial,
        joins_left,
        joins_right,
    }
}

fn arabic_form(ch: char) -> Option<ArabicForm> {
    Some(match ch {
        // ICU ArabicShaping(8) leaves bare hamza (and FE80) as U+0621 rather than inventing a
        // presentation-form glyph for it.
        '\u{0621}' => form('\u{0621}', '\u{0621}', '\u{0621}', '\u{0621}', false, false),
        '\u{0622}' => form('\u{fe81}', '\u{fe82}', '\u{fe81}', '\u{fe82}', true, false),
        '\u{0623}' => form('\u{fe83}', '\u{fe84}', '\u{fe83}', '\u{fe84}', true, false),
        '\u{0624}' => form('\u{fe85}', '\u{fe86}', '\u{fe85}', '\u{fe86}', true, false),
        '\u{0625}' => form('\u{fe87}', '\u{fe88}', '\u{fe87}', '\u{fe88}', true, false),
        '\u{0626}' => form('\u{fe89}', '\u{fe8a}', '\u{fe8b}', '\u{fe8c}', true, true),
        '\u{0627}' => form('\u{fe8d}', '\u{fe8e}', '\u{fe8d}', '\u{fe8e}', true, false),
        '\u{0628}' => form('\u{fe8f}', '\u{fe90}', '\u{fe91}', '\u{fe92}', true, true),
        '\u{0629}' => form('\u{fe93}', '\u{fe94}', '\u{fe93}', '\u{fe94}', true, false),
        '\u{062a}' => form('\u{fe95}', '\u{fe96}', '\u{fe97}', '\u{fe98}', true, true),
        '\u{062b}' => form('\u{fe99}', '\u{fe9a}', '\u{fe9b}', '\u{fe9c}', true, true),
        '\u{062c}' => form('\u{fe9d}', '\u{fe9e}', '\u{fe9f}', '\u{fea0}', true, true),
        '\u{062d}' => form('\u{fea1}', '\u{fea2}', '\u{fea3}', '\u{fea4}', true, true),
        '\u{062e}' => form('\u{fea5}', '\u{fea6}', '\u{fea7}', '\u{fea8}', true, true),
        '\u{062f}' => form('\u{fea9}', '\u{feaa}', '\u{fea9}', '\u{feaa}', true, false),
        '\u{0630}' => form('\u{feab}', '\u{feac}', '\u{feab}', '\u{feac}', true, false),
        '\u{0631}' => form('\u{fead}', '\u{feae}', '\u{fead}', '\u{feae}', true, false),
        '\u{0632}' => form('\u{feaf}', '\u{feb0}', '\u{feaf}', '\u{feb0}', true, false),
        '\u{0633}' => form('\u{feb1}', '\u{feb2}', '\u{feb3}', '\u{feb4}', true, true),
        '\u{0634}' => form('\u{feb5}', '\u{feb6}', '\u{feb7}', '\u{feb8}', true, true),
        '\u{0635}' => form('\u{feb9}', '\u{feba}', '\u{febb}', '\u{febc}', true, true),
        '\u{0636}' => form('\u{febd}', '\u{febe}', '\u{febf}', '\u{fec0}', true, true),
        '\u{0637}' => form('\u{fec1}', '\u{fec2}', '\u{fec3}', '\u{fec4}', true, true),
        '\u{0638}' => form('\u{fec5}', '\u{fec6}', '\u{fec7}', '\u{fec8}', true, true),
        '\u{0639}' => form('\u{fec9}', '\u{feca}', '\u{fecb}', '\u{fecc}', true, true),
        '\u{063a}' => form('\u{fecd}', '\u{fece}', '\u{fecf}', '\u{fed0}', true, true),
        '\u{0641}' => form('\u{fed1}', '\u{fed2}', '\u{fed3}', '\u{fed4}', true, true),
        '\u{0642}' => form('\u{fed5}', '\u{fed6}', '\u{fed7}', '\u{fed8}', true, true),
        '\u{0643}' => form('\u{fed9}', '\u{feda}', '\u{fedb}', '\u{fedc}', true, true),
        '\u{0644}' => form('\u{fedd}', '\u{fede}', '\u{fedf}', '\u{fee0}', true, true),
        '\u{0645}' => form('\u{fee1}', '\u{fee2}', '\u{fee3}', '\u{fee4}', true, true),
        '\u{0646}' => form('\u{fee5}', '\u{fee6}', '\u{fee7}', '\u{fee8}', true, true),
        '\u{0647}' => form('\u{fee9}', '\u{feea}', '\u{feeb}', '\u{feec}', true, true),
        '\u{0648}' => form('\u{feed}', '\u{feee}', '\u{feed}', '\u{feee}', true, false),
        '\u{0649}' => form('\u{feef}', '\u{fef0}', '\u{feef}', '\u{fef0}', true, false),
        '\u{064a}' => form('\u{fef1}', '\u{fef2}', '\u{fef3}', '\u{fef4}', true, true),
        '\u{0640}' => form('\u{0640}', '\u{0640}', '\u{0640}', '\u{0640}', true, true),
        _ => return None,
    })
}

/// ICU 51.2 accepts already-shaped Arabic presentation forms and first collapses their contextual
/// variants back to the base character before applying LETTERS_SHAPE.  The four lam-alef
/// ligature pairs intentionally stay presentation forms: they are terminal outputs, not ordinary
/// joining letters in the input normalization path.
fn unshape_arabic_char(ch: char) -> char {
    match ch {
        // ICU 51.2's shaping table treats these four Arabic combining code points as the
        // lam-alef ligature selectors used by the old renderer (a historical ICU quirk, but part
        // of the 1.8.9 observable path).
        '\u{065c}' => '\u{fef5}',
        '\u{065d}' => '\u{fef7}',
        '\u{065e}' => '\u{fef9}',
        '\u{065f}' => '\u{fefb}',
        '\u{fe70}' | '\u{fe71}' => '\u{064b}',
        '\u{fe72}' | '\u{fe73}' => '\u{064c}',
        '\u{fe74}' | '\u{fe75}' => '\u{064d}',
        '\u{fe76}' | '\u{fe77}' => '\u{064e}',
        '\u{fe78}' | '\u{fe79}' => '\u{064f}',
        '\u{fe7a}' | '\u{fe7b}' => '\u{0650}',
        '\u{fe7c}' | '\u{fe7d}' => '\u{0651}',
        '\u{fe7e}' | '\u{fe7f}' => '\u{0652}',
        '\u{fe80}' => '\u{0621}',
        '\u{fe81}' | '\u{fe82}' => '\u{0622}',
        '\u{fe83}' | '\u{fe84}' => '\u{0623}',
        '\u{fe85}' | '\u{fe86}' => '\u{0624}',
        '\u{fe87}' | '\u{fe88}' => '\u{0625}',
        '\u{fe89}' | '\u{fe8a}' | '\u{fe8b}' | '\u{fe8c}' => '\u{0626}',
        '\u{fe8d}' | '\u{fe8e}' => '\u{0627}',
        '\u{fe8f}' | '\u{fe90}' | '\u{fe91}' | '\u{fe92}' => '\u{0628}',
        '\u{fe93}' | '\u{fe94}' => '\u{0629}',
        '\u{fe95}' | '\u{fe96}' | '\u{fe97}' | '\u{fe98}' => '\u{062a}',
        '\u{fe99}' | '\u{fe9a}' | '\u{fe9b}' | '\u{fe9c}' => '\u{062b}',
        '\u{fe9d}' | '\u{fe9e}' | '\u{fe9f}' | '\u{fea0}' => '\u{062c}',
        '\u{fea1}' | '\u{fea2}' | '\u{fea3}' | '\u{fea4}' => '\u{062d}',
        '\u{fea5}' | '\u{fea6}' | '\u{fea7}' | '\u{fea8}' => '\u{062e}',
        '\u{fea9}' | '\u{feaa}' => '\u{062f}',
        '\u{feab}' | '\u{feac}' => '\u{0630}',
        '\u{fead}' | '\u{feae}' => '\u{0631}',
        '\u{feaf}' | '\u{feb0}' => '\u{0632}',
        '\u{feb1}' | '\u{feb2}' | '\u{feb3}' | '\u{feb4}' => '\u{0633}',
        '\u{feb5}' | '\u{feb6}' | '\u{feb7}' | '\u{feb8}' => '\u{0634}',
        '\u{feb9}' | '\u{feba}' | '\u{febb}' | '\u{febc}' => '\u{0635}',
        '\u{febd}' | '\u{febe}' | '\u{febf}' | '\u{fec0}' => '\u{0636}',
        '\u{fec1}' | '\u{fec2}' | '\u{fec3}' | '\u{fec4}' => '\u{0637}',
        '\u{fec5}' | '\u{fec6}' | '\u{fec7}' | '\u{fec8}' => '\u{0638}',
        '\u{fec9}' | '\u{feca}' | '\u{fecb}' | '\u{fecc}' => '\u{0639}',
        '\u{fecd}' | '\u{fece}' | '\u{fecf}' | '\u{fed0}' => '\u{063a}',
        '\u{fed1}' | '\u{fed2}' | '\u{fed3}' | '\u{fed4}' => '\u{0641}',
        '\u{fed5}' | '\u{fed6}' | '\u{fed7}' | '\u{fed8}' => '\u{0642}',
        '\u{fed9}' | '\u{feda}' | '\u{fedb}' | '\u{fedc}' => '\u{0643}',
        '\u{fedd}' | '\u{fede}' | '\u{fedf}' | '\u{fee0}' => '\u{0644}',
        '\u{fee1}' | '\u{fee2}' | '\u{fee3}' | '\u{fee4}' => '\u{0645}',
        '\u{fee5}' | '\u{fee6}' | '\u{fee7}' | '\u{fee8}' => '\u{0646}',
        '\u{fee9}' | '\u{feea}' | '\u{feeb}' | '\u{feec}' => '\u{0647}',
        '\u{feed}' | '\u{feee}' => '\u{0648}',
        '\u{feef}' | '\u{fef0}' => '\u{0649}',
        '\u{fef1}' | '\u{fef2}' | '\u{fef3}' | '\u{fef4}' => '\u{064a}',
        '\u{fef6}' => '\u{fef5}',
        '\u{fef8}' => '\u{fef7}',
        '\u{fefa}' => '\u{fef9}',
        '\u{fefc}' => '\u{fefb}',
        _ => ch,
    }
}

fn arabic_ligature_selector(ch: char) -> Option<char> {
    Some(match ch {
        '\u{065c}' => '\u{fef6}',
        '\u{065d}' => '\u{fef8}',
        '\u{065e}' => '\u{fefa}',
        '\u{065f}' => '\u{fefc}',
        _ => return None,
    })
}

fn isolated_arabic_alef(ch: char) -> Option<char> {
    Some(match ch {
        '\u{0622}' => '\u{fe81}',
        '\u{0623}' => '\u{fe83}',
        '\u{0625}' => '\u{fe87}',
        '\u{0627}' => '\u{fe8d}',
        _ => return None,
    })
}

fn arabic_mark(ch: char) -> Option<(char, char)> {
    Some(match ch {
        '\u{064b}' => ('\u{fe70}', '\u{fe71}'),
        '\u{064c}' => ('\u{fe72}', '\u{fe73}'),
        '\u{064d}' => ('\u{fe74}', '\u{fe75}'),
        '\u{064e}' => ('\u{fe76}', '\u{fe77}'),
        '\u{064f}' => ('\u{fe78}', '\u{fe79}'),
        '\u{0650}' => ('\u{fe7a}', '\u{fe7b}'),
        '\u{0651}' => ('\u{fe7c}', '\u{fe7d}'),
        '\u{0652}' => ('\u{fe7e}', '\u{fe7f}'),
        _ => return None,
    })
}

fn is_arabic_mark(ch: char) -> bool {
    arabic_mark(ch).is_some() || matches!(ch, '\u{0670}')
}

fn previous_joining_index(chars: &[char], index: usize) -> Option<usize> {
    let mut cursor = index;
    while cursor > 0 {
        cursor -= 1;
        match chars[cursor] {
            ch if is_arabic_mark(ch) => {}
            '\u{200d}' => {}
            '\u{200c}' => return None,
            _ => return Some(cursor),
        }
    }
    None
}

fn next_joining_index(chars: &[char], index: usize) -> Option<usize> {
    let mut cursor = index + 1;
    while cursor < chars.len() {
        match chars[cursor] {
            ch if is_arabic_mark(ch) => cursor += 1,
            '\u{200d}' => cursor += 1,
            '\u{200c}' => return None,
            _ => return Some(cursor),
        }
    }
    None
}

fn has_zwnj_between(chars: &[char], start: usize, end: usize) -> bool {
    chars[start + 1..end].contains(&'\u{200c}')
}

fn mark_is_medial(chars: &[char], index: usize) -> bool {
    let Some(previous) = previous_joining_index(chars, index) else {
        return false;
    };
    let Some(next) = next_joining_index(chars, index) else {
        return false;
    };
    if has_zwnj_between(chars, previous, next) {
        return false;
    }
    // ICU does not use a medial vowel mark inside a lam-alef ligature: the ligature consumes the
    // two letters as one joining cluster. A ZWJ intentionally suppresses the ligature.
    if chars[previous] == '\u{0644}'
        && matches!(
            chars[next],
            '\u{0622}' | '\u{0623}' | '\u{0625}' | '\u{0627}'
        )
        && !chars[index + 1..next].contains(&'\u{200d}')
    {
        return false;
    }
    arabic_form(chars[previous]).is_some_and(|left| left.joins_right)
        && arabic_form(chars[next]).is_some()
}

fn shape_arabic(text: &str) -> String {
    let raw_chars: Vec<char> = text.chars().collect();
    let chars: Vec<char> = raw_chars.iter().copied().map(unshape_arabic_char).collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];

        if let Some((isolated, medial)) = arabic_mark(ch) {
            out.push(if mark_is_medial(&chars, index) {
                medial
            } else {
                isolated
            });
            index += 1;
            continue;
        }

        // Lam-alef ligatures are formed only when no ZWNJ/ZWJ intervenes. Combining marks may
        // occur between the two letters and are retained before the ligature in logical order.
        if ch == '\u{0644}' {
            // ICU 51.2 has a legacy selector path for U+065C..U+065F: in a direct
            // lam-selector-alef cluster it emits a final lam form, a final lam-alef selector
            // ligature, and the original alef as an isolated presentation form. These selectors
            // are otherwise normalized to their isolated ligature forms above.
            if let (Some(selector), Some(alef)) = (
                raw_chars
                    .get(index + 1)
                    .copied()
                    .and_then(arabic_ligature_selector),
                raw_chars
                    .get(index + 2)
                    .copied()
                    .and_then(isolated_arabic_alef),
            ) {
                let joined = previous_joining_index(&chars, index)
                    .and_then(|previous| arabic_form(chars[previous]))
                    .is_some_and(|form| form.joins_right);
                out.push(if joined { '\u{fee0}' } else { '\u{fedf}' });
                out.push(selector);
                out.push(alef);
                index += 3;
                continue;
            }
            if let Some(next) = next_joining_index(&chars, index) {
                let ligature = match chars[next] {
                    '\u{0627}' => Some(('\u{fefb}', '\u{fefc}')),
                    '\u{0623}' => Some(('\u{fef7}', '\u{fef8}')),
                    '\u{0625}' => Some(('\u{fef9}', '\u{fefa}')),
                    '\u{0622}' => Some(('\u{fef5}', '\u{fef6}')),
                    _ => None,
                };
                if let Some((isolated, final_form)) = ligature {
                    let has_zwj = chars[index + 1..next].contains(&'\u{200d}');
                    let has_zwnj = chars[index + 1..next].contains(&'\u{200c}');
                    if !has_zwj && !has_zwnj {
                        let joined = previous_joining_index(&chars, index)
                            .and_then(|previous| arabic_form(chars[previous]))
                            .is_some_and(|form| form.joins_right);
                        out.push(if joined { final_form } else { isolated });
                        // ICU emits the lam-alef ligature before any combining marks that were
                        // between the two logical letters. Bidi then carries those NSM marks
                        // with the ligature in the visual run.
                        for mark in &chars[index + 1..next] {
                            if let Some((isolated, medial)) = arabic_mark(*mark) {
                                out.push(if mark_is_medial(&chars, index + 1) {
                                    medial
                                } else {
                                    isolated
                                });
                            } else {
                                out.push(*mark);
                            }
                        }
                        index = next + 1;
                        continue;
                    }
                }
            }
        }

        let Some(current) = arabic_form(ch) else {
            out.push(ch);
            index += 1;
            continue;
        };
        let previous_index = previous_joining_index(&chars, index);
        let next_index = next_joining_index(&chars, index);
        let zwj_to_next =
            next_index.is_some_and(|next| chars[index + 1..next].contains(&'\u{200d}'));
        let zwj_from_previous =
            previous_index.is_some_and(|previous| chars[previous + 1..index].contains(&'\u{200d}'));
        let previous = previous_index
            .and_then(|i| arabic_form(chars[i]))
            .is_some_and(|form| form.joins_right)
            && previous_index.is_some_and(|i| !has_zwnj_between(&chars, i, index));
        let next = next_index
            .and_then(|i| arabic_form(chars[i]))
            .is_some_and(|form| form.joins_left)
            && next_index.is_some_and(|i| !has_zwnj_between(&chars, index, i));
        let shaped = if zwj_to_next && ch == '\u{0644}' {
            current.initial
        } else if zwj_from_previous
            && matches!(ch, '\u{0622}' | '\u{0623}' | '\u{0625}' | '\u{0627}')
        {
            current.final_form
        } else {
            match (previous && current.joins_left, next && current.joins_right) {
                (true, true) => current.medial,
                (true, false) => current.final_form,
                (false, true) => current.initial,
                (false, false) => current.isolated,
            }
        };
        out.push(shaped);
        index += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_mixed_hebrew_latin_numbers_and_mirroring_vectors() {
        assert_eq!(
            reorder("abc \u{05d0}\u{05d1}\u{05d2}!"),
            "abc \u{05d2}\u{05d1}\u{05d0}!"
        );
        assert_eq!(
            reorder("\u{05e9}\u{05dc}\u{05d5}\u{05dd} abc!"),
            "!abc \u{05dd}\u{05d5}\u{05dc}\u{05e9}"
        );
        assert_eq!(
            reorder("abc \u{05e9}\u{05dc}\u{05d5}\u{05dd} 123"),
            "abc 123 \u{05dd}\u{05d5}\u{05dc}\u{05e9}"
        );
        assert_eq!(
            reorder("123 \u{05e9}\u{05dc}\u{05d5}\u{05dd} 456"),
            "456 \u{05dd}\u{05d5}\u{05dc}\u{05e9} 123"
        );
        assert_eq!(reorder("!@#"), "#@!");
        assert_eq!(reorder("\u{05d0}(abc)"), "(abc)\u{05d0}");
        assert_eq!(
            reorder("\u{0661}\u{0662}\u{0663} \u{05d0}"),
            "\u{05d0} \u{0661}\u{0662}\u{0663}"
        );
        assert_eq!(
            reorder("\u{05d0} \u{0661}\u{0662}\u{0663}"),
            "\u{0661}\u{0662}\u{0663} \u{05d0}"
        );
    }

    #[test]
    fn java_embedding_controls_are_retained_and_reordered() {
        assert_eq!(
            reorder("abc\u{202b}\u{05d0}\u{05d1}\u{202c}xyz"),
            "abc\u{05d1}\u{05d0}\u{202b}\u{202c}xyz"
        );
        assert_eq!(
            reorder("abc\u{202e}123\u{202c}xyz"),
            "abc321\u{202e}\u{202c}xyz"
        );
        assert_eq!(
            reorder("abc\u{2067}\u{05d0}\u{05d1}\u{2069}xyz"),
            "abc\u{05d1}\u{05d0}\u{2067}\u{2069}xyz"
        );
        assert_eq!(
            reorder("\u{05d0}\u{05d1}\u{202b}abc\u{202c}xyz"),
            "\u{202b}abc\u{202c}xyz\u{05d1}\u{05d0}"
        );
    }

    #[test]
    fn java_arabic_shaping_vectors_cover_marks_joins_and_ligatures() {
        assert_eq!(
            reorder("\u{0633}\u{0644}\u{0627}\u{0645}"),
            "\u{fee1}\u{fefc}\u{feb3}"
        );
        assert_eq!(reorder("\u{0644}\u{064e}\u{0627}"), "\u{fe76}\u{fefb}");
        assert_eq!(
            reorder("\u{0628}\u{200c}\u{062d}"),
            "\u{fea1}\u{200c}\u{fe8f}"
        );
        assert_eq!(
            reorder("\u{0628}\u{200d}\u{062d}"),
            "\u{fea2}\u{200d}\u{fe91}"
        );
        assert_eq!(
            reorder("\u{0644}\u{200d}\u{0627}"),
            "\u{fe8e}\u{200d}\u{fedf}"
        );
    }

    #[test]
    fn java_combining_marks_follow_their_bidi_level() {
        assert_eq!(
            reorder("\u{05d0}\u{05b0}\u{05d1}"),
            "\u{05d1}\u{05b0}\u{05d0}"
        );
        assert_eq!(
            reorder("\u{0634}\u{064e}\u{062f}\u{0651}\u{0629}"),
            "\u{fe93}\u{fe7c}\u{feaa}\u{fe77}\u{feb7}"
        );
    }

    #[test]
    fn java_legacy_controls_and_presentation_forms() {
        assert_eq!(reorder("a\u{0000}\u{05d0}"), "a\u{05d0}\u{0000}");
        assert_eq!(
            reorder("a\u{0000}\u{05d0}\u{0001}c"),
            "a\u{05d0}\u{0000}\u{0001}c"
        );
        assert_eq!(
            reorder("a\u{00a7}c\u{05d0}\u{00a7}r"),
            "a\u{00a7}c\u{05d0}\u{00a7}r"
        );
        assert_eq!(reorder("a\u{202a}\u{05d0}"), "a\u{05d0}\u{202a}");
        assert_eq!(reorder("a\u{2067}\u{05d0}"), "a\u{05d0}\u{2067}");
        assert_eq!(reorder("\u{0621}"), "\u{0621}");
        assert_eq!(reorder("\u{fe80}"), "\u{0621}");
        assert_eq!(reorder("\u{fe82}"), "\u{fe81}");
        assert_eq!(reorder("\u{fefc}"), "\u{fefb}");
        assert_eq!(reorder("\u{0644}\u{064e}\u{0627}"), "\u{fe76}\u{fefb}");
        assert_eq!(
            reorder("\u{0644}\u{065c}\u{0627}"),
            "\u{fe8d}\u{fef6}\u{fedf}"
        );
        assert_eq!(
            reorder("\u{0644}\u{065d}\u{0627}"),
            "\u{fe8d}\u{fef8}\u{fedf}"
        );
        assert_eq!(
            reorder("\u{0644}\u{065e}\u{0627}"),
            "\u{fe8d}\u{fefa}\u{fedf}"
        );
        assert_eq!(
            reorder("\u{0644}\u{065f}\u{0627}"),
            "\u{fe8d}\u{fefc}\u{fedf}"
        );
    }

    #[test]
    fn java_non_bmp_directional_data_controls_and_numbers() {
        assert_eq!(reorder("a\u{10800}\u{05d0}"), "a\u{05d0}\u{10800}");
        assert_eq!(reorder("a\u{1e800}1"), "a1\u{1e800}");
        assert_eq!(reorder("a\u{1ee00}\u{05d0}"), "a\u{05d0}\u{1ee00}");
        assert_eq!(reorder("a\u{e007e}\u{05d0}"), "a\u{05d0}\u{e007e}");
        assert_eq!(reorder("a\u{1d7ce}\u{05d0}"), "a\u{1d7ce}\u{05d0}");
        assert_eq!(reorder("a\u{1f100}\u{05d0}"), "a\u{1f100}\u{05d0}");
    }

    #[test]
    fn java_icu51_supplementary_class_boundaries_remain_stable() {
        // These are the representative supplementary classes whose ICU 51 data is called out
        // by the FontRenderer parity audit. Keep the boundary points explicit so a future table
        // cleanup cannot silently collapse them into the generic LTR fallback.
        let cases = [
            ('\u{107ff}', BidiClass::LeftToRight),
            ('\u{10800}', BidiClass::RightToLeft),
            ('\u{10e60}', BidiClass::ArabicNumber),
            ('\u{1d167}', BidiClass::NonspacingMark),
            ('\u{1d173}', BidiClass::BoundaryNeutral),
            ('\u{1d7ce}', BidiClass::EuropeanNumber),
            ('\u{1f100}', BidiClass::EuropeanNumber),
            ('\u{1f300}', BidiClass::OtherNeutral),
            ('\u{e0100}', BidiClass::NonspacingMark),
        ];
        for (ch, expected) in cases {
            assert_eq!(bidi_class(ch), expected, "U+{:04X}", ch as u32);
        }

        // The class boundary is observable through FontRenderer's visual order: an RTL
        // supplementary script must stay with the Hebrew run, while a supplementary number
        // remains in the LTR run.
        assert_eq!(reorder("a\u{10800}\u{05d0}"), "a\u{05d0}\u{10800}");
        assert_eq!(reorder("a\u{10e60}\u{05d0}"), "a\u{05d0}\u{10e60}");
    }
}
