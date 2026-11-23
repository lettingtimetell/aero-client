//! Vanilla textures for the loading scenes, read straight from the 1.8.9 jar.
//!
//! The scenes draw while the game's own asset load is still running, so they cannot use the
//! game's resource manager or atlas. Like the title screen's greenhouse
//! (`crate::menu_scene`), they read the vanilla jar privately: never the player's resource
//! packs, never the game's registries. A handful of small PNGs, decoded once.
//!
//! [`JarPrefetch::start`] reads them on a background thread, so the client can start it before
//! the window exists and the first loading frame finds them ready.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};

/// One decoded texture: tightly packed RGBA8, straight (not premultiplied) alpha, exactly as the
/// PNG stores it. An animated texture is the whole vertical strip.
#[derive(Clone, PartialEq, Eq)]
pub struct JarImage {
    /// `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Pixels.
    pub width: u32,
    /// Pixels.
    pub height: u32,
}

impl std::fmt::Debug for JarImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JarImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl JarImage {
    /// The texel at `(x, y)` (clamped), RGBA8.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }
}

/// The textures one scene asked for, by path (relative to `assets/minecraft/`, for example
/// `"textures/blocks/tallgrass.png"`). A path the jar lacks, or a jar that could not be opened,
/// simply has no entry: every scene must still draw something sensible without them.
#[derive(Debug, Clone, Default)]
pub struct JarTextures {
    images: Vec<(String, JarImage)>,
    /// Why nothing could be read, when the jar itself failed (for the log).
    error: Option<String>,
}

impl JarTextures {
    /// No textures (no jar, or a test).
    #[must_use]
    pub fn empty() -> JarTextures {
        JarTextures::default()
    }

    /// Read and decode `paths` from the vanilla jar at `jar`. Blocking (a few milliseconds for a
    /// dozen 16x16 PNGs; most of it is reading the jar's central directory). Never fails: a
    /// missing jar or texture is logged and left out.
    #[must_use]
    pub fn load(jar: &Path, paths: &[&str]) -> JarTextures {
        let started = std::time::Instant::now();
        let pack = match ZipResourcePack::open(jar) {
            Ok(pack) => pack,
            Err(error) => {
                let message = format!("{}: {error}", jar.display());
                tracing::warn!(%message, "loading screen: the vanilla jar could not be opened");
                return JarTextures {
                    images: Vec::new(),
                    error: Some(message),
                };
            }
        };
        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(pack));
        let mut images = Vec::with_capacity(paths.len());
        for path in paths {
            let location = ResourceLocation::new("minecraft", *path);
            match resources.load_png_rgba(&location) {
                Ok((rgba, width, height, _metadata)) => {
                    images.push((
                        (*path).to_owned(),
                        JarImage {
                            rgba,
                            width,
                            height,
                        },
                    ));
                }
                Err(error) => {
                    tracing::warn!(%error, path, "loading screen: a vanilla texture is missing");
                }
            }
        }
        tracing::debug!(
            textures = images.len(),
            ms = started.elapsed().as_secs_f64() * 1000.0,
            "loading screen: vanilla textures read from the jar"
        );
        JarTextures {
            images,
            error: None,
        }
    }

    /// The texture at `path` (relative to `assets/minecraft/`), if it was read.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&JarImage> {
        self.images
            .iter()
            .find(|(name, _)| name == path)
            .map(|(_, image)| image)
    }

    /// How many textures were read.
    #[must_use]
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether nothing was read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Why the jar could not be opened, if it could not.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// [`JarTextures::load`] on a background thread.
#[derive(Debug)]
pub struct JarPrefetch {
    handle: Option<std::thread::JoinHandle<JarTextures>>,
}

impl JarPrefetch {
    /// Start reading `paths` from `jar` (or nothing, without a jar). Returns at once.
    #[must_use]
    pub fn start(jar: Option<PathBuf>, paths: &'static [&'static str]) -> JarPrefetch {
        let Some(jar) = jar else {
            return JarPrefetch { handle: None };
        };
        // A reader that panics leaves the loading screen without its jar
        // textures (see `take`); contained, so that is not the game's crash report either.
        let handle = std::thread::Builder::new()
            .name("loading-scene-jar".to_owned())
            .spawn(move || {
                mc_core::panic_scope::contain(|| JarTextures::load(&jar, paths)).unwrap_or_default()
            })
            .map_err(|error| {
                tracing::warn!(%error, "loading screen: could not start the jar reader");
            })
            .ok();
        JarPrefetch { handle }
    }

    /// The textures, waiting for the reader if it has not finished (it normally has: it starts
    /// before the window is created). Empty when there was no jar or the reader panicked.
    #[must_use]
    pub fn take(mut self) -> JarTextures {
        self.handle
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default()
    }

    /// Whether the reader has finished (always true without a jar).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_jar_gives_no_textures_and_says_why() {
        let textures = JarTextures::load(
            Path::new("definitely/not/a/real/1.8.9.jar"),
            &["textures/blocks/tallgrass.png"],
        );
        assert!(textures.is_empty());
        assert!(textures.error().is_some());
        assert!(textures.get("textures/blocks/tallgrass.png").is_none());
        assert!(JarPrefetch::start(None, &[]).take().is_empty());
    }

    #[test]
    fn texel_lookup_clamps() {
        let image = JarImage {
            rgba: vec![1, 2, 3, 4, 5, 6, 7, 8],
            width: 2,
            height: 1,
        };
        assert_eq!(image.texel(0, 0), [1, 2, 3, 4]);
        assert_eq!(image.texel(9, 9), [5, 6, 7, 8]);
    }
}
