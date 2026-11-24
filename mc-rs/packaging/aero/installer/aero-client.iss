; Aero Client Setup: installs the portable package for the current Windows user.
;
; It installs exactly what the portable zip's "Aero Client" folder holds (Aero Client.exe,
; README.txt and app\) into a folder named "Aero Client", without an administrator prompt:
; %LOCALAPPDATA%\Programs\Aero Client, or on a first install another folder the player chooses (one
; they can change without administrator rights). It adds an "Aero Client" shortcut to the Start
; menu (and, if ticked, the desktop) that starts the game itself, app\mc-client.exe, so the running
; game groups with a pinned shortcut. The client's updater (mc-rs/crates/mc-update) recognises an
; installed copy by that same layout and updates it the way it updates an unzipped one: it swaps
; app\ and the files next to it by renames and never touches anything else here (the uninstaller
; files, game\). So every release must keep the shortcuts' icon in app\ (app\aero-client.ico): an
; update replaces that folder.
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
; Never change it: a different one installs a second copy instead of upgrading. The client keeps
; that entry's version in step after an in-game update, when it starts itself; keep the two in step.
#define AppIdGuid "33D5C40D-DF3B-44A8-89C8-B29ABEFBF5B1"
; What portable players double-click; it starts ClientExe.
#define StarterExe "Aero Client.exe"
; The game. Started with no arguments it finds the Minecraft files and its folders on its own.
#define ClientExe "app\mc-client.exe"
#define IconFile "aero-client.ico"

; The package must be complete, and its updater marker must name the version this Setup claims
; (Setup writes it to Settings > Apps; the client reads the marker).
#define Package Copy(SourceDir, Len(SourceDir), 1) == "\" ? Copy(SourceDir, 1, Len(SourceDir) - 1) : SourceDir
#if !FileExists(Package + "\" + StarterExe) || !FileExists(Package + "\README.txt") || \
    !FileExists(Package + "\" + ClientExe) || !FileExists(Package + "\app\" + IconFile)
  #error The package is incomplete: SourceDir needs Aero Client.exe, README.txt, app\mc-client.exe and app\aero-client.ico
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
; For this Windows user only: no administrator prompt, and always a folder named "Aero Client"
; that this user can change without administrator rights (the in-client updater writes there), so
; the uninstaller only ever deletes inside such a folder (see NextButtonClick and
; PrepareToInstall). The folder page shows on a first install; an upgrade keeps the folder.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DisableDirPage=auto
UsePreviousAppDir=yes
; Browse adds "Aero Client" to the folder picked; NextButtonClick does the same for a typed one,
; a drive root included, so the root itself is never the install folder.
AppendDefaultDirName=yes
AllowRootDirectory=yes
AllowUNCPath=no
; A folder that already holds a portable copy is fine (game\ is kept), so Setup's "folder already
; exists" question would only confuse.
DirExistsWarning=no
DisableProgramGroupPage=yes
DisableWelcomePage=yes
; The tasks page's button says Install: folder (first install only), tasks, progress, done.
DisableReadyPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=Aero-Client-Setup
SetupIconFile={#Package}\app\{#IconFile}
UninstallDisplayIcon={app}\app\{#IconFile}
; No version in the name: after an in-game update the client refreshes the version shown.
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
; A running client is closed (after asking) before its files are replaced. Starting it again is
; left to the player (Setup's last page offers it).
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
; The in-client updater writes to the folder, so it must be one this Windows user can change.
SelectDirBrowseLabel=To continue, click Next. To choose another folder, click Browse. {#AppName} updates itself in this folder, so choose one you can change without administrator rights.
; Inno Setup's own text also suggests a network path, which AllowUNCPath=no refuses.
InvalidPath=You must enter a full path with a drive letter; for example:%n%nD:\Games\{#AppName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; The old app\ goes first, so an upgrade leaves exactly the new package (an update may have added
; files Setup never knew). Skipped while a client still runs from it: then Setup stops at that
; file and asks to retry, instead of leaving half a folder.
Type: filesandordirs; Name: "{app}\app"; Check: not ClientIsRunning
; Earlier packages started the game through this script; the package has none any more.
Type: files; Name: "{app}\Play Aero Client.cmd"

[Files]
; What the updater swaps, nothing else (crates\mc-update\src\stage.rs extracts the same).
Source: "{#Package}\{#StarterExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Package}\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Package}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Both start the game itself with no arguments, so a shortcut pinned to the taskbar and the running
; game share one taskbar button. No "minimized" flag: the game's window would start minimized.
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#ClientExe}"; WorkingDir: "{app}"; \
  IconFilename: "{app}\app\{#IconFile}"; Comment: "Play {#AppName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ClientExe}"; WorkingDir: "{app}"; \
  IconFilename: "{app}\app\{#IconFile}"; Comment: "Play {#AppName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ClientExe}"; WorkingDir: "{app}"; Description: "Launch {#AppName}"; \
  Flags: postinstall nowait skipifsilent

[UninstallDelete]
; What the updater swapped in or keeps (crates\mc-update\src\install.rs): Setup did not install
; those files, so it would not know to delete them. game\ is the player's (see the [Code]).
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\update"
Type: dirifempty; Name: "{app}"

[Code]
const
  FolderNeedsAdmin = 'Choose a folder you can change without administrator rights, such as one in your user folder or on another drive. {#AppName} updates itself in that folder.';

// Whether a program runs from this file: Windows refuses to open a running program for writing
// (the updater checks the same way). Opening it changes nothing.
function FileInUse(const Path: String): Boolean;
var
  Stream: TFileStream;
begin
  Result := False;
  if not FileExists(Path) then
    Exit;
  try
    Stream := TFileStream.Create(Path, fmOpenReadWrite or fmShareDenyNone);
    Stream.Free;
  except
    Result := True;
  end;
end;

// Whether the game runs from this install's app\mc-client.exe.
function ClientIsRunning(): Boolean;
begin
  Result := FileInUse(ExpandConstant('{app}\{#ClientExe}'));
end;

// Whether the full path Path is Folder or inside it.
function IsInside(const Path, Folder: String): Boolean;
begin
  Result := Pos(Uppercase(AddBackslash(Folder)), Uppercase(AddBackslash(Path))) = 1;
end;

// Whether this Windows user can change the folder Dir without administrator rights, as the
// in-client updater must: never inside Program Files or Windows (a Setup run as administrator
// could write there, the updater could not), and Setup can make a test folder with a test file in
// the nearest folder of Dir that exists (Dir itself when it exists). Both are removed again.
function UserCanChange(const Dir: String): Boolean;
var
  Existing, Parent, Probe: String;
begin
  Result := False;
  if IsInside(Dir, ExpandConstant('{commonpf64}')) or IsInside(Dir, ExpandConstant('{commonpf32}')) or
      IsInside(Dir, ExpandConstant('{win}')) then
    Exit;
  Existing := RemoveBackslashUnlessRoot(Dir);
  while not DirExists(Existing) do
  begin
    Parent := ExtractFileDir(Existing);
    if (Parent = '') or (CompareText(Parent, Existing) = 0) then
      Exit;
    Existing := Parent;
  end;
  Probe := AddBackslash(Existing) + 'aero-setup-test-' + GetDateTimeString('yyyymmddhhnnsszzz', #0, #0);
  if not CreateDir(Probe) then
    Exit;
  Result := SaveStringToFile(Probe + '\test.tmp', 'test', False);
  DeleteFile(Probe + '\test.tmp');
  RemoveDir(Probe);
end;

// The folder page (a first install): the folder must be named "Aero Client", and a typed folder
// gets that name added the way Browse adds it; then it must be one this Windows user can change.
// A silent install skips this: PrepareToInstall refuses a wrong /DIR= there.
function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir, Final: String;
begin
  Result := True;
  if (CurPageID <> wpSelectDir) or WizardSilent() then
    Exit;
  Dir := WizardDirValue();
  Final := Dir;
  if CompareText(ExtractFileName(Dir), '{#AppName}') <> 0 then
    Final := AddBackslash(Dir) + '{#AppName}';
  if not UserCanChange(Final) then
  begin
    MsgBox(FolderNeedsAdmin, mbError, MB_OK);
    Result := False;
  end
  else if Final <> Dir then
  begin
    // Shown in the box, so the player sees the folder before installing into it.
    WizardForm.DirEdit.Text := Final;
    MsgBox('{#AppName} installs into a folder of its own:' + #13#10#13#10 + Final + #13#10#13#10 +
      'Click Next to install it there, or choose another folder.', mbInformation, MB_OK);
    Result := False;
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

// The folder, however it was given (the folder page, an upgrade, /DIR= on the command line): only
// one named "Aero Client", since the uninstaller deletes app\, update\ and (when the player agrees)
// game\ inside it, and only one this Windows user can change, where updates can be installed.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Dir: String;
begin
  Result := '';
  Dir := RemoveBackslashUnlessRoot(ExpandConstant('{app}'));
  if CompareText(ExtractFileName(Dir), '{#AppName}') <> 0 then
    Result := '{#AppName} installs only into a folder named "{#AppName}".'
  else if not UserCanChange(Dir) then
    Result := FolderNeedsAdmin;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  // The game, or Aero Client.exe while it shows a message: their files could not be removed.
  while ClientIsRunning() or FileInUse(ExpandConstant('{app}\{#StarterExe}')) do
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
