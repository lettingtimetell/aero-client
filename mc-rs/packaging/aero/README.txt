AERO CLIENT
===========

A Minecraft 1.8.9 PvP client. You need your own copy of Minecraft Java Edition.


ONE-TIME SETUP (about 5 minutes)
--------------------------------
Aero Client uses the Minecraft 1.8.9 game files on your own PC, and you sign in inside it with
the Microsoft account that owns Minecraft.

  1. Install the official Minecraft Launcher (from minecraft.net) and sign in to it with the
     Microsoft account that owns Minecraft Java Edition.
  2. In the launcher, click "Installations" > "New installation", choose the version
     "release 1.8.9", and click "Create". Press "Play" on it once, and close the game when it
     opens. (This downloads the game files that Aero Client uses.)


PLAY
----
  3. Right-click the Aero Client zip and choose "Extract All...".
  4. Open the "Aero Client" folder and double-click "Play Aero Client".

  The first time, Windows may warn you, because Aero Client is not from a big company:
    - A blue "Windows protected your PC" box: click "More info", then "Run anyway".
    - A "Do you want to run this file?" box: untick "Always ask before opening this file",
      then click "Run".

  5. Sign in, once: on the title screen, click "Not signed in" at the top right (or open
     Options > "Accounts..."), then "Add account" and "Sign in with Microsoft". Aero Client
     shows a code: click "Open page", enter the code on Microsoft's page and sign in with the
     Microsoft account that owns Minecraft. Aero Client remembers the account for your Windows
     user on this PC, so next time you are already signed in.


THE INSTALLED VERSION (Aero-Client-Setup.exe)
---------------------------------------------
If you installed Aero Client with Aero-Client-Setup.exe instead of extracting the zip:
  - Start it from "Aero Client" in the Start menu, or the desktop shortcut. Skip steps 3 and 4;
    everything else is the same. Windows may show the first-time warnings above for the setup.
  - It is in the folder  %LOCALAPPDATA%\Programs\Aero Client  (paste that into the address bar
    of File Explorer). Its "game" folder holds your settings, screenshots and replays.
  - It updates itself like the zip does (see "Updates" below). Running a newer
    Aero-Client-Setup.exe also updates it and keeps your settings.
  - To remove it, open Settings > Apps, find Aero Client and choose Uninstall. It asks whether
    to delete your settings, screenshots and replays too: choose No to keep them for later.
    Your saved sign-in stays on this PC for your Windows user (remove the account on the
    Accounts screen first if you want it gone), and so do videos in Videos\Aero Client.


GOOD TO KNOW
------------
  - Right Shift in game opens the mod menu (FPS, keystrokes, zoom, crosshair, shaders and more),
    and so does the "Mods" button on the title screen and the pause menu. Type to search it.
    To use another key, click "Keybinds" on the left of the menu, then the top row.
  - Replays: turn on "Replay" in the mod menu to record your games, and watch them with the
    "Replays" button on the title screen. Videos you export go to your Videos\Aero Client folder.
  - Your settings, screenshots, replays and logs are in the "game" folder next to this file.
  - Texture packs go in  game\resourcepacks  (it appears after the first start).
  - Accounts: add, switch or remove them on the Accounts screen. Your sign-in is kept for your
    Windows user, outside the "Aero Client" folder, so a copy of the folder never contains it.
  - To move Aero Client, move the whole "Aero Client" folder.
  - Updates: when a new Aero Client is out, the title screen says so at the bottom left. An
    update never changes your "game" folder. To stop checking, open the mod menu and turn off
    Aero Title Screen > Check for updates.
  - When you join a server, Aero Client tells it its name ("Aero", where the original game says
    "vanilla").


IF SOMETHING GOES WRONG
-----------------------
  "Aero Client uses your own Minecraft 1.8.9 game files, but couldn't find them"
      Do setup steps 1 and 2 above. If your game files are in another folder (for example a
      launcher that keeps them somewhere else), click OK and select that folder: the one that
      contains the "assets" folder.

  "You're not signed in" when you join a server
      Do step 5 above.

  "The saved sign-in ... has expired" or "Sign in again to use it"
      Your saved sign-in has run out or can't be opened on this PC. Open the Accounts screen,
      click "Sign in again" and sign in once more.

  "Microsoft sign-in isn't set up in this build of Aero Client"
      This copy was built without sign-in. Get Aero Client from its official download page.

  The game closed with an error
      Send the two files in  game\logs  named latest-launch.log and latest-launch.errors.log
      to whoever gave you Aero Client.

  Nothing happens at all
      Your antivirus may have blocked it. Allow the "Aero Client" folder, extract the zip
      again (or run Aero-Client-Setup.exe again), and retry.


Not an official Minecraft product. Not approved by or associated with Mojang or Microsoft.
Contains no Minecraft game files; it uses the ones on your own PC. See app\NOTICE.txt.
