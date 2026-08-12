# pdf-perceptual-compare

`pdf-perceptual-compare` compares two PDFs page by page according to how their
rendered pages look, rather than by comparing PDF bytes or raw image pixels.
It is useful when PDFs may differ internally (metadata, font embedding,
compression, or generation tooling) but should render equivalently.

For each matching page, the tool renders both PDFs as RGB PNGs with Poppler,
then measures global and local Structural Similarity (SSIM). It exits with
status `0` when every page passes and `1` when one or more pages fail, making
it suitable for CI.

## Requirements

- Python 3.10 or newer
- Poppler utilities: `pdfinfo` and `pdftoppm`

For example, on Debian or Ubuntu:

```bash
sudo apt install poppler-utils
```

## Installation

Install the package in the current environment:

```bash
python -m pip install .
```

For development, install the test and lint dependencies too:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```bash
pdf-perceptual-compare original.pdf candidate.pdf
```

The command checks that both inputs exist and contain the same number of pages.
It renders every page at 150 DPI, compares corresponding pages, prints one
result line per page, and returns a non-zero status if any comparison fails.
Different page counts fail immediately.

## Common workflows

### Produce a report and investigate failures

```bash
pdf-perceptual-compare \
  original.pdf \
  candidate.pdf \
  --json artifacts/report.json \
  --save-failures artifacts/failures
```

`report.json` contains all page measurements and the effective parameters. For
every failed page, `artifacts/failures/page-NNNN/` contains:

- `diff-amplified.png` — an RGB absolute-difference image with differences
  amplified eightfold for inspection.
- `ssim-map.png` — a grayscale similarity map: white is highly similar and
  darker areas are less similar.

### Retain the rendered inputs

Rendered PNGs are normally kept in a temporary directory and removed on exit.
Use `--keep-rendered` to retain the exact images that were compared:

```bash
pdf-perceptual-compare before.pdf after.pdf \
  --dpi 200 \
  --keep-rendered artifacts/rendered
```

The directory receives `original-0001.png`, `candidate-0001.png`, and so on.

### Allow small translation differences

When a generator produces an otherwise equivalent page a few pixels away from
the reference, search for the best integer translation:

```bash
pdf-perceptual-compare original.pdf candidate.pdf --align 2
```

This tries every horizontal and vertical shift from -2 to +2 pixels and scores
only the overlapping region. Use it deliberately: it can hide small layout
movements that may matter to your application.

## Command-line arguments

The two positional arguments are required:

| Argument | Description |
| --- | --- |
| `original` | Reference PDF. |
| `candidate` | PDF to evaluate against the reference. |

### Rendering and performance

| Option | Default | Description |
| --- | ---: | --- |
| `--dpi DPI` | `150` | Resolution used to render each PDF page. Higher values detect smaller visual changes but use more CPU, memory, and temporary disk space. |
| `--jobs N` | CPU count | Maximum concurrent render or comparison operations. `N` must be at least 1. Lower it on memory-constrained machines. |
| `--align N` | `0` | Search integer translations up to `N` pixels in each direction before calculating metrics. `0` disables alignment. |

### Metrics and thresholds

| Option | Default | Description |
| --- | ---: | --- |
| `--tile PIXELS` | `128` | Edge length of the image tiles used for local SSIM statistics. Smaller tiles localize defects more precisely; larger tiles make statistics less sensitive to small regions. |
| `--blur RADIUS` | `0.5` | Gaussian blur radius, in rendered pixels, used for the blurred SSIM pass. Set to `0` to disable blur. |
| `--ssim SCORE` | `0.995` | Minimum raw global SSIM. |
| `--ssim-blur SCORE` | `0.999` | Minimum global SSIM after both pages are blurred. |
| `--local-p01 SCORE` | `0.980` | Minimum first-percentile tile SSIM for the raw pass. |
| `--local-threshold SCORE` | `0.980` | A tile whose mean SSIM is below this value is counted as suspicious. |
| `--max-bad-fraction FRACTION` | `0.005` | Largest permitted fraction of suspicious tiles. `0.005` means 0.5%. |

### Output and artifacts

| Option | Description |
| --- | --- |
| `--json PATH` | Write the full machine-readable result report to `PATH`. Parent directories are created when needed. |
| `--save-failures DIRECTORY` | Write diagnostics below one `page-NNNN` directory for each failed page. |
| `--keep-rendered DIRECTORY` | Copy all rendered original and candidate page PNGs to this directory. |

## How a page passes

An exactly equal rendered RGB image is reported as `PASS` with all similarity
scores equal to `1.0`. Otherwise, the page passes if either condition is true:

1. **Raw pass:** global SSIM is at least `--ssim`, the first percentile of tile
   SSIM is at least `--local-p01`, and the suspicious-tile fraction is at most
   `--max-bad-fraction`.
2. **Blurred pass:** blurred global SSIM is at least `--ssim-blur`, and the
   same suspicious-tile-fraction limit is met.

The blurred pass tolerates very small rasterization or anti-aliasing variation,
while the local bad-tile limit prevents a good global score from masking a
localized visual defect. A page whose rendered dimensions differ is reported
as `FAIL(size)`.

The defaults are conservative starting points, not universal correctness
criteria. Calibrate DPI, tile size, and thresholds using known-good and
known-bad examples from your own documents before making the result a release
or CI gate.

## Reading terminal output

Each result line has this form:

```text
   3  PASS       SSIM=0.999420  blur=0.999910  p01=0.986200  min=0.978100  bad= 0.000%
```

| Field | Meaning |
| --- | --- |
| `PASS`, `FAIL`, or `FAIL(size)` | Per-page verdict. Failed lines are red when standard output is an interactive terminal. |
| `SSIM` | Raw global similarity over the page (or aligned overlap). Higher is more similar. |
| `blur` | Global SSIM after Gaussian blur. A strong score here but weaker raw SSIM often indicates minor edge or anti-aliasing variation. |
| `p01` | First percentile of tile mean SSIM values. It reflects weaker local regions without relying on one worst tile. |
| `min` | Lowest tile mean SSIM; useful for locating an isolated worst region. |
| `bad` | Percentage of tiles below `--local-threshold`. |
| `shift=+X,+Y` | Printed only when alignment selected a non-zero translation. |

Scores closer to `1` indicate greater similarity. Read the fields together: a
high global SSIM with low `p01`, low `min`, or high `bad` usually means a small
area changed; low global and local scores point to a broader visual difference.
Inspect diagnostics and retained rendered images before changing thresholds.

## JSON report

Passing `--json report.json` writes a report after comparison completes. The
`results` array is ordered by completion internally, so consumers should use
each result's `page` field rather than assuming array order.

```json
{
  "original": "original.pdf",
  "candidate": "candidate.pdf",
  "pages": 2,
  "parameters": {
    "dpi": 150,
    "jobs": 8,
    "tile": 128,
    "blur": 0.5,
    "align": 0,
    "ssim": 0.995,
    "ssim_blur": 0.999,
    "local_p01": 0.98,
    "local_threshold": 0.98,
    "max_bad_fraction": 0.005
  },
  "summary": {
    "identical_pages": 1,
    "passed_pages": 2,
    "failed_pages": 0
  },
  "results": [
    {
      "page": 1,
      "identical": true,
      "shift_x": 0,
      "shift_y": 0,
      "ssim": 1.0,
      "ssim_blur": 1.0,
      "local_p01": 1.0,
      "local_min": 1.0,
      "local_bad_fraction": 0.0,
      "verdict": "PASS"
    }
  ]
}
```

| JSON field | Interpretation |
| --- | --- |
| `original`, `candidate` | Input paths as supplied to the command. |
| `pages` | Common page count that was compared. |
| `parameters` | Effective rendering, concurrency, alignment, and comparison settings. Preserve these with a report so scores remain reproducible. |
| `summary.identical_pages` | Number of pages whose rendered RGB pixels were exactly equal. |
| `summary.passed_pages` / `failed_pages` | Counts based on each page's `verdict`. |
| `results` | One object per compared page. `shift_x` and `shift_y` are the selected candidate translation in pixels. |

`verdict` is `PASS`, `FAIL`, or `FAIL(size)`. Treat a report as successful only
when `summary.failed_pages` is zero. If PDFs have different page counts, the
command exits with status `1` before rendering and does not write a JSON report.
