<p align="center"><img src="assets/icon.png" width="96" alt="TTPF Studio icon"></p>

<h1 align="center">TTPF Studio</h1>

<p align="center">
  A desktop tool for looking inside the data of <b>Grand Theft Auto VI</b>:<br>
  3D models, textures, the map, videos and game data, in one window.
</p>

<p align="center">
  <a href="https://github.com/woreksami/ttpf-studio/releases/latest"><b>Download for Windows</b></a>
  &nbsp;·&nbsp; <a href="#getting-started">Getting started</a>
  &nbsp;·&nbsp; <a href="#what-you-can-do">What you can do</a>
  &nbsp;·&nbsp; <a href="FORMATS.md">File formats</a>
</p>

---

## Getting started

1. Download **`TTPF-Studio-windows.zip`** from the [latest release](https://github.com/woreksami/ttpf-studio/releases/latest) and extract it anywhere.
2. Run **`TTPF Studio.exe`**. Nothing else to install.
3. On the welcome page, click **Open your game folder** and pick the folder that holds `levels.rpf`, `packs.rpf` and `data`.
4. Click **🧊 3D models → Vehicles**.

Want a shortcut? Double-click **`Add desktop shortcut.bat`** in the same folder; it adds the app to your Desktop and Start menu.

> **"Windows protected your PC"?** The app isn't code-signed, so Windows warns about any new download. Click **More info → Run anyway**.

**First run:** each archive is scanned once, on every CPU core, and remembered. The biggest one (`levels.rpf`, 71 GB) takes a few minutes; after that everything opens in seconds.

## What you can do

### 3D models

Pick a category (**Vehicles, Animals, People, Clothing, Weapons, Props, Plants, Buildings, Terrain**) and click any model to see it.

- **Vehicles come out complete.** The game's own vehicle data supplies each car's body colour, its rims and tyres, and pickup beds. Complete high-detail vehicles are listed first, marked ★.
- **Real textures and lighting.** The view runs on your graphics card with full-resolution textures, glossy paint and smooth edges, and stays fluid while you turn the model.
- **Controls:** drag to turn, right-drag to move, mouse wheel to zoom, double-click to reset.
- **Export to OBJ** with materials and textures, ready for Blender. Export one model or a whole list.

### The map

Open an archive that contains terrain and click **Export the map**. You get a single file (glTF or OBJ) with the terrain and every building and prop placed where the game puts them, at real height, plus a top-down colour image of the ground. Export one 512 m cell, a block of cells, or the whole map.

### Textures

Open any `.ttpf` texture pack to browse its textures: zoom, pan, look at single colour channels, flip between a viewer and a thumbnail grid. Export to PNG or DDS, or replace a texture and save a modded copy of the pack (the original is never touched). All common GPU formats are supported (BC1–BC7, RGBA8, RGBA16F), and packs with a damaged header still open.

### The Library

**Library → Update index** catalogues the whole game folder once. Then search everything by name (models, textures, videos, game data) from one box, sorted into a tree: *3D models › Vehicles › banshee*, *Textures › Peds & clothing*, *Videos*, *Game data › CVehicleModelInfo*… Double-click a 3D model to open it in the 3D view.

### Archives, videos and game data

`.rpf` archives are opened as far as their unencrypted contents allow: XML and text files, game-data resources and Bink videos. Videos can be played or converted to MP4 (needs the free [RAD Video Tools](https://www.radgametools.com/bnkdown.htm), and [ffmpeg](https://ffmpeg.org/) for MP4).

## Good to know

| | |
|---|---|
| **Supported version** | GTA VI for Xbox Series X\|S. The PC and PS5 versions are untested. |
| **File names** | The archive file list is encrypted and this tool doesn't decrypt it, so recovered files are named from their content. Model names come from the models themselves. |
| **No OpenGL?** | The 3D view falls back to a built-in software renderer (slower, same picture). |
| **Your files** | The game files are only read. The app writes to the folders you choose and to `%APPDATA%\TTPF Studio` (settings, caches, error log). |

**Not done yet:** character skin and clothing colours, vehicle tuning parts and per-part paint layers, textures stored inside model files, skeletons and animation.

## Running from source

Python 3.10 or newer:

```powershell
git clone https://github.com/woreksami/ttpf-studio
cd ttpf-studio
pip install -r requirements.txt
python ttpf_tool.py
```

`python ttpf_tool.py --install` copies the app to `%LOCALAPPDATA%\Programs\TTPF Studio` and adds shortcuts.

Command line:

```powershell
python ttpf_tool.py --dump <packs folder> <output>   # export every texture (add --rpf for archive contents)
python ttpf_tool.py --rpf <archive.rpf>              # print an archive header
python ttpf_tool.py --selftest                       # run the built-in tests
python ttpf_tool.py --glcheck                        # check that the GPU 3D view works on this PC
```

## Credits

The decoders in this tool are its own code. Format knowledge came from these projects, with thanks:

- [6Stars](https://github.com/Ahmet-Dedeler/6Stars): model layout
- [RageForge](https://github.com/nullbrik/RageForge): map placement
- [SixNG](https://github.com/holdmysocks/SixNG): vehicle data, skeletons and vertex streams
- [bcpack-rs](https://github.com/Ahmet-Dedeler/bcpack-rs): the texture chunk table layout

## Legal

An unofficial fan project, not affiliated with or endorsed by Rockstar Games or Take-Two Interactive. Grand Theft Auto, GTA and GTA VI are trademarks of Take-Two Interactive. This repository contains no game files, game assets or encryption keys; use the tool only with files you're allowed to use.

MIT licence, see [LICENSE](LICENSE).
