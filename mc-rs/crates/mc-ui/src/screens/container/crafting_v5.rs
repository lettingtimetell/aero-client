//! Minecraft 1.7.10's crafting-grid prediction: the result `ContainerPlayer` and
//! `ContainerWorkbench` show when their grid changes (java17: `onCraftMatrixChanged`, which asks
//! `CraftingManager.findMatchingRecipe`), from 1.7.10's own recipe list and special recipes
//! ([`mc_data::v5::find_matching_recipe_v5`]). A child of `model_v5`, for [`Container`]'s private
//! state; `model.rs` reaches it from one line in `recompute_craft_result`.
//!
//! A vanilla server never sends the result slot while the grid changes (it is a `SlotCrafting`),
//! so this prediction is what the player sees, and what the next click on the result sends as
//! `clicked_item`.

use mc_data::v5::{find_matching_recipe_v5, RecipeTransformV5, SlotTagV5, TagEntryV5};
use mc_nbt::TagType;

use super::super::{all_registered, Container, PredictedCraft};
use crate::screens::container::stack_ext::ItemStack;

impl Container {
    /// java17: `ContainerPlayer.onCraftMatrixChanged` / `ContainerWorkbench.onCraftMatrixChanged`,
    /// which always write the manager's result, nothing included: a result a server wrote into
    /// the slot (a window resync) is replaced by the next grid change, as in 1.7.10. (1.8.9 mode
    /// keeps a server's result when its table finds nothing; 1.7.10's manager is complete here.)
    /// As in 1.8.9 mode, a stack the item registry does not know leaves the result slot alone.
    pub(in crate::screens::container::model) fn recompute_craft_result_v5(&mut self) {
        let Some((start, end, _, _)) = self.crafting_grid_shape() else {
            return;
        };
        if !all_registered(self.version, &self.items[start..end]) {
            return;
        }
        let Some((slots, width, height)) = self.crafting_grid() else {
            return;
        };
        let grid = mc_data::CraftingGrid::new(width, height, &slots)
            .expect("the container's fixed crafting shape is valid");
        let tags: Vec<SlotTagV5> = self.items[start..end]
            .iter()
            .map(|stack| self.slot_tag_v5(stack.as_ref()))
            .collect();
        match find_matching_recipe_v5(grid, &tags) {
            Some(result) => {
                let output = Self::wire_stack(result.stack);
                let output = match &result.transform {
                    RecipeTransformV5::Shared(transform) => {
                        self.apply_dynamic_transform(start, output, transform)
                    }
                    RecipeTransformV5::CopyTag { source_slot } => {
                        self.copied_stack(start, *source_slot, output)
                    }
                };
                self.items[0] = Some(output);
                // Provenance only: the 1.7.10 pickup (`model_v5.rs`) reads no recipe kind.
                self.predicted_craft = Some(PredictedCraft::Vanilla {
                    kind: None,
                    transform: mc_data::DynamicRecipeTransform::Plain,
                });
            }
            None => {
                self.predicted_craft = None;
                self.items[0] = None;
            }
        }
    }

    /// What 1.7.10's special recipes read from one grid stack besides its item, count and
    /// damage. The leather colour and the map scale are read as 1.8.9 mode reads them (java17
    /// `ItemArmor.hasColor`/`getColor` and `ItemMap.getMapData` agree with java18's).
    fn slot_tag_v5(&self, stack: Option<&ItemStack>) -> SlotTagV5 {
        let Some(stack) = stack else {
            return SlotTagV5::default();
        };
        let shared = self.dynamic_slot_data(Some(stack));
        let root = stack.nbt.as_ref();
        let entry = |key: &str| match root.and_then(|root| root.get(key)) {
            None => TagEntryV5::Absent,
            Some(tag) if tag.tag_type() == TagType::Compound => TagEntryV5::Compound,
            Some(_) => TagEntryV5::NotCompound,
        };
        SlotTagV5 {
            has_tag: root.is_some(),
            leather_color: shared.leather_color,
            display_not_compound: entry("display") == TagEntryV5::NotCompound,
            explosion: entry("Explosion"),
            map_scale: shared.map_scale,
        }
    }
}
