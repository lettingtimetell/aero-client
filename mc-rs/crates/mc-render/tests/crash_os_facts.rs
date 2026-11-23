//! A crash report's local time and operating system now come from the OS's own calls
//! (`mc_render::w21_mb`) instead of three `powershell.exe` runs per report. The report must say
//! exactly what it said before: on this PC, every value must equal what the replaced PowerShell
//! query prints (the queries are copied here verbatim from the old crash-report code), for the
//! current moment and two earlier ones (standard and daylight time). Windows only; skips loudly
//! without PowerShell.

#![cfg(windows)]

use std::time::{SystemTime, UNIX_EPOCH};

use mc_render::w21_mb;

/// The first output line of a PowerShell command, as the old crash-report code read it.
fn powershell(script: &str) -> Option<String> {
    let output = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|line| line.trim().to_owned())
        .filter(|line| !line.is_empty())
}

#[test]
fn crash_report_facts_equal_the_powershell_queries_they_replace() {
    let Some(os) = powershell(
        "$os = Get-CimInstance Win32_OperatingSystem; \"$($os.Caption)|$($os.Version)\"",
    ) else {
        eprintln!("SKIP: PowerShell or WMI is not available here");
        return;
    };
    let (caption, version) = os.split_once('|').expect("caption|version");
    assert_eq!(w21_mb::windows_caption().as_deref(), Some(caption.trim()));
    assert_eq!(w21_mb::windows_version().as_deref(), Some(version.trim()));

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("after 1970")
        .as_secs() as i64;
    // Now, and three and six months back (the other side of a daylight saving change for most
    // of the year), at odd hours and minutes that exercise padding and AM/PM.
    let moments = [
        now,
        now - 91 * 86_400 - (3 * 3_600 + 17 * 60 + 53),
        now - 182 * 86_400 + (7 * 3_600 + 42 * 60 + 11),
    ];
    for seconds in moments {
        let fields = powershell(&format!(
            "$date=[DateTimeOffset]::FromUnixTimeSeconds({seconds}).ToLocalTime(); \
             $date.ToString('yyyy|MM|dd|HH|mm|ss',[Globalization.CultureInfo]::InvariantCulture)"
        ))
        .expect("PowerShell's local time");
        let [year, month, day, hour, minute, second] =
            w21_mb::local_time_fields(seconds).expect("the OS's local time");
        assert_eq!(
            format!("{year:04}|{month:02}|{day:02}|{hour:02}|{minute:02}|{second:02}"),
            fields,
            "local time of {seconds}"
        );

        let short = powershell(&format!(
            "$culture=[Globalization.CultureInfo]::CurrentCulture; \
             $datePattern=$culture.DateTimeFormat.ShortDatePattern.Replace('yyyy','yy'); \
             $date=[DateTimeOffset]::FromUnixTimeSeconds({seconds}).ToLocalTime(); \
             $date.ToString($datePattern+' '+$culture.DateTimeFormat.ShortTimePattern,$culture)"
        ))
        .expect("PowerShell's short date and time");
        assert_eq!(
            w21_mb::local_short_date_time(seconds).as_deref(),
            Some(short.as_str()),
            "short date and time of {seconds}"
        );
        eprintln!("{seconds}: {fields} / {short}");
    }
    eprintln!("operating system: {caption} version {version}");
}
