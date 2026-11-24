//! One string value of an existing key under `HKEY_CURRENT_USER`: what the packaged start reads
//! and keeps in step in the Settings > Apps entry `Aero-Client-Setup.exe` writes.
//!
//! A key is never created here: setting a value of a missing key is an error that writes
//! nothing, so a portable copy (no Setup entry) leaves the registry as it was.

use std::io;

/// The `REG_SZ` value `name` of `HKEY_CURRENT_USER\<key>`; `Ok(None)` when the key or the value
/// doesn't exist.
///
/// # Errors
///
/// The value isn't a string, isn't valid UTF-16, or can't be read (and on other platforms).
pub fn read_current_user_string(key: &str, name: &str) -> io::Result<Option<String>> {
    #[cfg(windows)]
    {
        crate::sys::read_current_user_string(key, name)
    }
    #[cfg(not(windows))]
    {
        let _ = (key, name);
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// Set the `REG_SZ` value `name` of the existing key `HKEY_CURRENT_USER\<key>` to `value`.
///
/// # Errors
///
/// The key doesn't exist ([`io::ErrorKind::NotFound`]; nothing is created) or can't be written
/// (and on other platforms).
pub fn set_current_user_string(key: &str, name: &str, value: &str) -> io::Result<()> {
    #[cfg(windows)]
    {
        crate::sys::set_current_user_string(key, name, value)
    }
    #[cfg(not(windows))]
    {
        let _ = (key, name, value);
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::process::Command;

    /// A temporary key under `HKEY_CURRENT_USER\Software`, removed on drop. Made and removed with
    /// Windows' own `reg.exe`, so the code under test is checked against a second implementation.
    struct TestKey(String);

    impl TestKey {
        fn new(tag: &str) -> TestKey {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_nanos());
            TestKey(format!(
                r"Software\AeroShellRegistryTest-{tag}-{}-{nanos}",
                std::process::id()
            ))
        }

        fn reg(&self, args: &[&str]) -> bool {
            let full = format!(r"HKCU\{}", self.0);
            let mut command = Command::new(reg_exe());
            command.arg(args[0]).arg(&full).args(&args[1..]);
            command.output().is_ok_and(|output| output.status.success())
        }

        fn create_with(&self, name: &str, value: &str) {
            assert!(
                self.reg(&["add", "/v", name, "/t", "REG_SZ", "/d", value, "/f"]),
                "reg add"
            );
        }

        fn exists(&self) -> bool {
            self.reg(&["query"])
        }
    }

    impl Drop for TestKey {
        fn drop(&mut self) {
            let _ = self.reg(&["delete", "/f"]);
        }
    }

    fn reg_exe() -> std::path::PathBuf {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
        std::path::Path::new(&root).join("System32").join("reg.exe")
    }

    #[test]
    fn a_value_is_read_and_set_only_in_a_key_that_exists() {
        let key = TestKey::new("rw");
        assert_eq!(
            read_current_user_string(&key.0, "DisplayVersion").unwrap(),
            None
        );
        // A missing key is never created.
        let error = set_current_user_string(&key.0, "DisplayVersion", "1.0.0").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound, "{error}");
        assert!(!key.exists(), "nothing was created");

        key.create_with("InstallLocation", r"C:\Users\Example\Aero Client\");
        assert_eq!(
            read_current_user_string(&key.0, "InstallLocation")
                .unwrap()
                .as_deref(),
            Some(r"C:\Users\Example\Aero Client\")
        );
        assert_eq!(
            read_current_user_string(&key.0, "DisplayVersion").unwrap(),
            None
        );
        set_current_user_string(&key.0, "DisplayVersion", "1.2.3").unwrap();
        assert_eq!(
            read_current_user_string(&key.0, "DisplayVersion")
                .unwrap()
                .as_deref(),
            Some("1.2.3")
        );
        // Longer than the first guess of any buffer, and not ASCII.
        let long = "\u{e9}".repeat(3000);
        set_current_user_string(&key.0, "Long", &long).unwrap();
        assert_eq!(
            read_current_user_string(&key.0, "Long").unwrap().as_deref(),
            Some(long.as_str())
        );
        set_current_user_string(&key.0, "Empty", "").unwrap();
        assert_eq!(
            read_current_user_string(&key.0, "Empty")
                .unwrap()
                .as_deref(),
            Some("")
        );
    }
}
