# burn
# ASCII Candle

A tiny flame starts a big chain reaction. Strike a match, light a candle, and watch it fall—setting the city skyline ablaze as a glowing figure rises from the fire.

This project includes a standalone browser animation and a Python/Tkinter version.

## Features

- **Browser animation:** An ASCII-inspired city skyline, spreading fire, rising smoke, and a glowing human figure.
- **Responsive canvas:** The animation scales to fit your browser window.
- **Python edition:** A standalone candle animation rendered in a Tkinter window.
- **GIF export:** Export the Python animation as an animated GIF.

> The city-fire and rising-figure effects are part of the browser version. The Python edition renders the candle animation.

## Play in your browser

Open [`index.html`](./index.html) in a modern browser. No build step or JavaScript dependencies are required.

| Input | Action |
| --- | --- |
| Click the animation | Restart |
| `R` | Restart |
| `Space` | Pause or resume |

## Run the Python version

Requires Python 3.8+ and Tkinter. Install NumPy and start the animation:

```bash
python -m pip install -r requirements.txt
python candle.py
```

| Key | Action |
| --- | --- |
| `R` | Restart |
| `Space` | Pause or resume |
| `Esc` | Quit |

Tkinter is included with most Windows and macOS Python installers. On Linux, install your distribution’s Tkinter package (often named `python3-tk`) if it is missing.

## Export a GIF

Install Pillow, then render the Python animation:

```bash
python -m pip install pillow
python candle.py --export candle.gif --seconds 14 --fps 20
```

## Project files

- [`index.html`](./index.html) — standalone browser animation.
- [`candle.py`](./candle.py) — Python/Tkinter animation and GIF exporter.
- [`requirements.txt`](./requirements.txt) — Python dependency list.
