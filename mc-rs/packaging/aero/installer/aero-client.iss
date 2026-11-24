; Aero Client Setup: installs the portable package for the current Windows user.
;
; It installs exactly what the portable zip's "Aero Client" folder holds (Play Aero Client.cmd,
; README.txt and app\) into %LOCALAPPDATA%\Programs\Aero Client, without an administrator prompt,
; and adds an "Aero Client" shortcut to the Start menu (and, if ticked, the desktop). The client's
; updater (mc-rs/crates/mc-update) recognises an installed copy by that same layout and updates it
; the way it updates an unzipped one: it swaps app\ and the two files next to it by renames and
; never touches anything else here (the uninstaller files, game\). So every release must keep the
; shortcuts' icon in app\ (app\aero-client.ico): an update replaces that folder.
;
; package-aero-client.ps1 builds it after the zip, as release\Aero-Client-Setup.exe:
;
;   ISCC.exe /DAppVersion=0.1.0 "/DSourceDir=<release\Aero Client>" "/DOutputDir=<release>"
;            "/DAssetDir=<folder made by make-installer-assets.ps1>" aero-client.iss
;
; Optional: /DSign with /Saerosign=<signtool command line ending in $f> signs Setup and its
; uninstaller. Needs Inno Setup 6.3 or later; the wizard pictures need 6.5.2, and following
; Windows' light or dark mode 6.6.
;
; Uninstalling removes the program, what the updater added (app\, update\), the shortcuts and the
; Settings > Apps entry. It keeps the player's game\ folder (settings, screenshots, replays)
; unless the player says yes to deleting it, and a silent uninstall always keeps it. Saved
; sign-ins live outside this folder (they are shared with portable copies) and are never touched.

#if Ver < EncodeVer(6, 3, 0)
  #error Aero Client's installer needs Inno Setup 6.3 or later
#endif
#ifndef AppVersion
  #error Pass the version being packaged: /DAppVersion=x.y.z
#endif
#ifndef SourceDir
  #error Pass the staged package: /DSourceDir=<the "Aero Client" folder that goes into the zip>
#endif
#ifndef OutputDir
  #define OutputDir SourcePath + "Output"
#endif

#define AppName "Aero Client"
; The Settings > Apps entry is HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\{<this>}_is1.
; Never change it: a different one installs a second copy instead of upgrading. The launcher
; (packaging\aero\launcher.ps1) keeps that entry's version in step after an update; keep in step.
#define AppIdGuid "33D5C40D-DF3B-44A8-89C8-B29ABEFBF5B1"
#define LauncherCmd "Play Aero Client.cmd"
#define IconFile "aero-client.ico"

; The package must be complete, and its updater marker must name the version this Setup claims
; (Setup writes it to Settings > Apps; the client reads the marker).
#define Package Copy(SourceDir, Len(SourceDir), 1) == "\" ? Copy(SourceDir, 1, Len(SourceDir) - 1) : SourceDir
#if !FileExists(Package + "\" + LauncherCmd) || !FileExists(Package + "\README.txt") || \
    !FileExists(Package + "\app\mc-client.exe") || !FileExists(Package + "\app\launcher.ps1") || \
    !FileExists(Package + "\app\" + IconFile)
  #error The package is incomplete: SourceDir needs Play Aero Client.cmd, README.txt, app\mc-client.exe, app\launcher.ps1 and app\aero-client.ico
#endif
#define MarkerVersionFound 0
#define MarkerHandle
#define MarkerLine
#sub ReadMarkerLine
  #define MarkerLine = FileRead(MarkerHandle)
  #if Pos('"version"', MarkerLine) > 0 && Pos('"' + AppVersion + '"', MarkerLine) > 0
    // An assignment, not a #define: a #define here would only last for this call.
    #expr MarkerVersionFound = 1
  #endif
#endsub
#for {MarkerHandle = FileOpen(Package + "\app\aero-install.json"); \
  MarkerHandle && !FileEof(MarkerHandle); ""} ReadMarkerLine
#if MarkerHandle
  #expr FileClose(MarkerHandle)
#endif
#if !MarkerVersionFound
  #error app\aero-install.json in the package does not name the version given by /DAppVersion
#endif

[Setup]
AppId={{{#AppIdGuid}}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
AppPublisherURL=https://aero-client.com/
AppSupportURL=https://aero-client.com/
AppUpdatesURL=https://aero-client.com/
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
VersionInfoProductTextVersion={#AppVersion}
; For this Windows user only: no administrator prompt, and a fixed folder (no folder page), so the
; uninstaller only ever deletes inside a folder named "Aero Client" (see PrepareToInstall).
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DisableDirPage=yes
UsePreviousAppDir=yes
DisableProgramGroupPage=yes
DisableWelcomePage=yes
; The tasks page's button says Install: tasks, progress, done.
DisableReadyPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=Aero-Client-Setup
SetupIconFile={#Package}\app\{#IconFile}
UninstallDisplayIcon={app}\app\{#IconFile}
; No version in the name: after an in-game update the launcher refreshes the version shown.
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
; A running client is closed (after asking) before its files are replaced. It never asks Windows
; to restart it: a client started without the launcher would not find the game files.
CloseApplications=yes
RestartApplications=no
SetupMutex=AeroClientSetup-{#AppIdGuid}
ShowLanguageDialog=no
#if Ver >= EncodeVer(6, 6, 0)
WizardStyle=modern dynamic
#else
WizardStyle=modern
#endif
#ifdef AssetDir
  #if Ver >= EncodeVer(6, 5, 2)
WizardSmallImageFile={#AssetDir}\wizard-small-*.png
WizardImageFile={#AssetDir}\wizard-large-*.png
WizardImageBackColor=#e8f3f9
    #if Ver >= EncodeVer(6, 6, 0)
WizardImageBackColorDynamicDark=#2e3a3f
    #endif
  #endif
#endif
#ifdef Sign
SignTool=aerosign
SignedUninstaller=yes
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
; The tasks page is the last one before installing (no Ready page), so its button says Install.
SelectTasksLabel2=Select the additional tasks you would like Setup to perform while installing [name], then click Install.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; The old app\ goes first, so an upgrade leaves exactly the new package (an update may have added
; files Setup never knew). Skipped while a client still runs from it: then Setup stops at that
; file and asks to retry, instead of leaving half a folder.
Type: filesandordirs; Name: "{app}\app"; Check: not ClientIsRunning

[Files]
; What the updater swaps, nothing else (crates\mc-update\src\stage.rs extracts the same three).
Source: "{#Package}\{#LauncherCmd}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Package}\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Package}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Both start Play Aero Client.cmd, which starts the launcher hidden; minimized, so no console
; window shows while it does.
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#LauncherCmd}"; WorkingDir: "{app}"; \
  IconFilename: "{app}\app\{#IconFile}"; Comment: "Play {#AppName}"; Flags: runminimized
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#LauncherCmd}"; WorkingDir: "{app}"; \
  IconFilename: "{app}\app\{#IconFile}"; Comment: "Play {#AppName}"; Tasks: desktopicon; \
  Flags: runminimized

[Run]
Filename: "{app}\{#LauncherCmd}"; WorkingDir: "{app}"; Description: "Launch {#AppName}"; \
  Flags: postinstall nowait skipifsilent runhidden

[UninstallDelete]
; What the updater swapped in or keeps (crates\mc-update\src\install.rs): Setup did not install
; those files, so it would not know to delete them. game\ is the player's (see the [Code]).
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\update"
Type: dirifempty; Name: "{app}"

[Code]
// Whether a client runs from this install's app\mc-client.exe: Windows refuses to open a
// running program for writing (the updater checks the same way). Opening it changes nothing.
function ClientIsRunning(): Boolean;
var
  Exe: String;
  Stream: TFileStream;
begin
  Result := False;
  Exe := ExpandConstant('{app}\app\mc-client.exe');
  if not FileExists(Exe) then
    Exit;
  try
    Stream := TFileStream.Create(Exe, fmOpenReadWrite or fmShareDenyNone);
    Stream.Free;
  except
    Result := True;
  end;
end;

// DisableReadyPage leaves the caption of the button on the page before it alone.
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpSelectTasks then
    WizardForm.NextButton.Caption := SetupMessage(msgButtonInstall)
  else if CurPageID = wpFinished then
    WizardForm.NextButton.Caption := SetupMessage(msgButtonFinish)
  else
    WizardForm.NextButton.Caption := SetupMessage(msgButtonNext);
end;

// Only a folder named "Aero Client": the uninstaller deletes app\, update\ and (when the player
// agrees) game\ inside it. The folder is fixed; this guards a /DIR= on the command line.
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if CompareText(ExtractFileName(RemoveBackslashUnlessRoot(ExpandConstant('{app}'))), '{#AppName}') <> 0 then
    Result := '{#AppName} installs only into a folder named "{#AppName}".';
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  while ClientIsRunning() do
  begin
    if SuppressibleMsgBox('{#AppName} is running. Close the game, then click Retry.',
        mbError, MB_RETRYCANCEL, IDCANCEL) <> IDRETRY then
    begin
      Result := False;
      Exit;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Game: String;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  Game := ExpandConstant('{app}\game');
  // A silent uninstall asks nobody, so it keeps the player's files.
  if not DirExists(Game) or UninstallSilent() then
    Exit;
  if MsgBox('Also delete your {#AppName} settings, screenshots and replays?' + #13#10#13#10 +
      'If you choose No, they stay in this folder:' + #13#10 + Game,
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
  begin
    if not DelTree(Game, True, True, True) then
      MsgBox('Some of your files could not be deleted. They are still in:' + #13#10 + Game,
        mbInformation, MB_OK);
  end;
end;
