//! Potion and enchantment ids in 1.7.10.
//!
//! Potion ids 1-23 are the same effects in both versions; only jump boost's colour differs
//! (java17: `Potion.jump`, `potion/Potion.java:29`, 7889559; java18: `potion/Potion.java:33`,
//! 2293580). Enchantment ids are the same apart from depth strider (id 8), which 1.8.9 added
//! (java18: `Enchantment.depthStrider`, `enchantment/Enchantment.java:35`, read by
//! `EntityLivingBase.moveEntityWithHeading` through `EnchantmentHelper.getDepthStriderModifier`):
//! a 1.7.10 client has no enchantment 8, so it neither changes water movement nor shows a tooltip
//! line for it (java17: `Enchantment.enchantmentsList` has no entry 8, and `ItemStack.getTooltip`
//! skips null entries, `item/ItemStack.java:535-537`).

use crate::misc::{enchantment, potion, EnchantmentInfo, PotionEffect};

/// Depth strider, 1.8.9 only.
const DEPTH_STRIDER: u8 = 8;

/// Jump boost's liquid colour in 1.7.10, `0x786297` (java17: `Potion.jump`,
/// `potion/Potion.java:29`); 1.8.9 uses `0x22FF4C`.
const JUMP_BOOST_COLOR_1_7_10: u32 = 0x78_62_97;

/// A potion effect as 1.7.10 defines it, `None` for ids it does not register.
#[must_use]
pub fn potion_v5(id: u8) -> Option<PotionEffect> {
    let mut effect = *potion(id)?;
    if id == 8 {
        effect.liquid_color = JUMP_BOOST_COLOR_1_7_10;
    }
    Some(effect)
}

/// An enchantment 1.7.10 registers, `None` for depth strider and every unregistered id.
#[inline]
#[must_use]
pub fn enchantment_v5(id: u8) -> Option<&'static EnchantmentInfo> {
    if id == DEPTH_STRIDER {
        return None;
    }
    enchantment(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::{enchantments, potions};

    #[test]
    fn potion_ids_and_colours() {
        for effect in potions() {
            let v5 = potion_v5(effect.id).expect("same ids 1-23");
            if effect.id == 8 {
                assert_eq!(v5.liquid_color, 0x78_62_97);
                assert_ne!(effect.liquid_color, v5.liquid_color, "1.8.9 differs");
            } else {
                assert_eq!(&v5, effect);
            }
        }
        assert_eq!(potions().count(), 23);
        assert!(potion_v5(0).is_none() && potion_v5(24).is_none());
    }

    #[test]
    fn enchantments_without_depth_strider() {
        assert!(enchantment(8).is_some(), "1.8.9 has depth strider");
        assert!(enchantment_v5(8).is_none());
        let v5: Vec<u8> = (0..=u8::MAX)
            .filter(|&id| enchantment_v5(id).is_some())
            .collect();
        assert_eq!(v5.len(), 24);
        assert_eq!(enchantments().count(), 25);
        assert!(std::ptr::eq(
            enchantment_v5(16).unwrap(),
            enchantment(16).unwrap()
        ));
    }
}
