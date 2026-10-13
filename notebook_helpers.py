"""
Helpers for `model_deployment.ipynb`.

They are kept out of the notebook so its cells stay focused on the
deployment ideas instead of on drawing code:

* `set_plot_style()`  - one consistent matplotlib look for every chart
* `show_file()`       - pretty-print a project file (predict.py, Dockerfile, ...)
* `BackgroundServer`  - run a Flask app in a thread so the notebook can call it
* `draw_*()`          - the concept diagrams (deployment stack, ports, Docker, ...)

Run `python notebook_helpers.py` to regenerate the PNGs in `images/`
(they are used by the README).
"""

import logging
import threading
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

# ---------------------------------------------------------------------------
# Palette (validated categorical order: blue, orange, aqua, yellow, ...)
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#6250d6', '#e34948'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# ink & chrome
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

# light tints used as box fills in the diagrams
BLUE_T, ORANGE_T, AQUA_T = '#dbe9fb', '#fbe1d5', '#d5f1e6'
YELLOW_T, VIOLET_T, GRAY_T, RED_T = '#fcedc6', '#e4e0f9', '#f0efec', '#f9dcdc'

# one colour per deployment layer - reused in every diagram
LAYER = {
    'model': (ORANGE_T, ORANGE),
    'web': (YELLOW_T, YELLOW),
    'env': (AQUA_T, AQUA),
    'docker': (BLUE_T, BLUE),
    'cloud': (VIOLET_T, VIOLET),
    'client': (GRAY_T, INK2),
}

MONO = 'DejaVu Sans Mono'
IMAGES_DIR = Path(__file__).parent / 'images'


def set_plot_style():
    """Apply a clean, consistent style to all matplotlib charts."""
    from cycler import cycler

    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'figure.dpi': 110,
        'axes.facecolor': SURFACE,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK2,
        'axes.titlecolor': INK,
        'axes.titlesize': 12.5,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.titlepad': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.prop_cycle': cycler(color=SERIES),
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2,
        'text.color': INK,
        'font.size': 10.5,
        'legend.frameon': False,
        'lines.linewidth': 2,
        'savefig.bbox': 'tight',
        'savefig.facecolor': SURFACE,
    })


# ---------------------------------------------------------------------------
# Notebook utilities
# ---------------------------------------------------------------------------
_LANGUAGES = {'.py': 'python', '.toml': 'toml', '.json': 'json', '.yml': 'yaml',
              '.txt': 'text', '.md': 'markdown'}


def show_file(path):
    """Display a project file as a syntax-highlighted block in the notebook."""
    from IPython.display import Markdown, display

    path = Path(path)
    if path.name == 'Dockerfile':
        language = 'dockerfile'
    elif path.name == 'Pipfile':
        language = 'toml'
    else:
        language = _LANGUAGES.get(path.suffix, '')

    content = path.read_text(encoding='utf-8').rstrip()
    display(Markdown(f'📄 **`{path.as_posix()}`**\n```{language}\n{content}\n```'))


class BackgroundServer:
    """Run a Flask (WSGI) app in a background thread.

    `app.run()` blocks the cell forever - and with debug=True its reloader
    tries to restart the Jupyter kernel process. Running the server in a thread
    keeps the notebook free, so the *next* cells can send requests to it.

    >>> server = BackgroundServer(app, port=9696).start()
    >>> ...  # requests.get('http://localhost:9696/ping')
    >>> server.stop()
    """

    def __init__(self, app, host='127.0.0.1', port=9696, quiet=False):
        self.app, self.host, self.port = app, host, port
        self.quiet = quiet
        self._server = None
        self._thread = None

    @property
    def url(self):
        return f'http://{self.host}:{self.port}'

    def set_quiet(self, quiet=True):
        """Hide (or show) the one-log-line-per-request output of the server."""
        self.quiet = quiet
        logging.getLogger('werkzeug').setLevel(logging.WARNING if quiet else logging.INFO)

    def start(self):
        from werkzeug.serving import make_server

        self.set_quiet(self.quiet)
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f'✅ Server running at {self.url}  (in a background thread)')
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = None
            print(f'🛑 Server at {self.url} stopped')

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# Diagram primitives
# ---------------------------------------------------------------------------
def _canvas(width, height, scale=0.85):
    fig, ax = plt.subplots(figsize=(width * scale, height * scale))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def _box(ax, x, y, w, h, fc=GRAY_T, ec=INK2, lw=1.6, ls='-', radius=0.14, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))


def _text(ax, x, y, s, size=10.5, weight='normal', color=INK, ha='center', va='center',
          family=None, zorder=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            family=family, zorder=zorder, linespacing=1.35)


def _arrow(ax, start, end, color=INK2, lw=1.8, ls='-', style='-|>', rad=0.0):
    ax.annotate('', xy=end, xytext=start, zorder=4,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=16,
                                connectionstyle=f'arc3,rad={rad}'))


def _titled_box(ax, x, y, w, h, title, subtitle=None, colors=LAYER['client'], ls='-',
                title_size=11.5, center=False):
    """A box with a bold title (and optional subtitle) at its top or centre."""
    fc, ec = colors
    _box(ax, x, y, w, h, fc=fc, ec=ec, ls=ls)
    if center:
        # centre the whole title + subtitle block vertically
        n_lines = subtitle.count('\n') + 1 if subtitle else 0
        block = 0.3 + (0.12 + 0.24 * n_lines if subtitle else 0)
        title_y = y + h / 2 + block / 2 - 0.15
        _text(ax, x + w / 2, title_y, title, size=title_size, weight='bold')
        if subtitle:
            _text(ax, x + w / 2, title_y - 0.27, subtitle, size=9.5, color=INK2, va='top')
    else:
        _text(ax, x + 0.25, y + h - 0.32, title, size=title_size, weight='bold', ha='left')
        if subtitle:
            _text(ax, x + 0.25, y + h - 0.66, subtitle, size=9.5, color=INK2, ha='left')


def _finish(fig, name, save, show):
    if save:
        IMAGES_DIR.mkdir(exist_ok=True)
        fig.savefig(IMAGES_DIR / f'{name}.png', dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def draw_deployment_stack(save=False, show=True):
    """The big picture: every deployment layer wraps the one inside it."""
    fig, ax = _canvas(12, 6.6)

    # client + request/response arrows
    _titled_box(ax, 0.2, 2.3, 2.4, 2.0, 'Client', 'an app, a website,\ncurl, a browser',
                center=True)
    _arrow(ax, (2.6, 3.7), (3.4, 3.7))
    _arrow(ax, (3.4, 2.9), (2.6, 2.9))
    _text(ax, 3.0, 4.1, 'request', size=9, color=INK2)
    _text(ax, 3.0, 2.5, 'response', size=9, color=INK2)

    layers = [
        ('Cloud — AWS Elastic Beanstalk',
         'rents an always-on computer with a public URL, load balancing & scaling', 'cloud'),
        ('Docker container — system dependencies',
         'Debian "slim" OS + Python 3.12 — identical on every machine', 'docker'),
        ('Python environment — Pipenv / requirements.txt',
         'exact library versions: scikit-learn==1.9.1, flask, gunicorn', 'env'),
        ('Web service — Flask app (predict.py) behind gunicorn',
         'turns the predict() function into the URL  POST /predict', 'web'),
        ('Model — model_C=1.0.bin',
         'DictVectorizer + LogisticRegression, loaded with pickle', 'model'),
    ]
    x, y, w, h = 3.4, 0.2, 8.4, 6.2
    for i, (title, subtitle, layer) in enumerate(layers):
        innermost = i == len(layers) - 1
        _titled_box(ax, x, y, w, h, title, subtitle, colors=LAYER[layer], center=innermost)
        x, y, w, h = x + 0.35, y + 0.35, w - 0.7, h - 1.2

    _finish(fig, 'deployment_stack', save, show)


def draw_request_response(save=False, show=True):
    """What travels between a client and our web service."""
    fig, ax = _canvas(12, 5.0)

    _titled_box(ax, 0.3, 0.7, 3.0, 3.6, 'Client', 'predict_client.py\nrequests · curl · browser',
                center=True)
    _titled_box(ax, 8.7, 0.7, 3.0, 3.6, 'Web service', 'Flask app  ·  predict.py',
                colors=LAYER['web'])
    _text(ax, 8.95, 2.45,
          '1. read the JSON body\n2. dv.transform(customer)\n3. model.predict_proba(X)\n'
          '4. return JSON', size=9.5, ha='left', family=MONO)

    # request
    _arrow(ax, (3.3, 3.75), (8.7, 3.75))
    _text(ax, 6.0, 4.05, '①  HTTP POST  http://localhost:9696/predict', size=10, weight='bold')
    _box(ax, 3.75, 2.85, 4.5, 0.6, fc='#ffffff', ec=AXIS, lw=1)
    _text(ax, 6.0, 3.15, '{"contract": "month-to-month", "tenure": 1, ...}', size=9,
          family=MONO)

    # response
    _box(ax, 3.75, 1.55, 4.5, 0.6, fc='#ffffff', ec=AXIS, lw=1)
    _text(ax, 6.0, 1.85, '{"churn": true, "churn_probability": 0.63}', size=9, family=MONO)
    _arrow(ax, (8.7, 1.2), (3.3, 1.2))
    _text(ax, 6.0, 0.9, '②  HTTP 200 OK  +  JSON body', size=10, weight='bold')

    _finish(fig, 'request_response', save, show)


def draw_venv_isolation(save=False, show=True):
    """Why two projects on one machine need separate environments."""
    fig, ax = _canvas(12, 5.6)

    # --- without ---
    _text(ax, 3.0, 5.25, 'Without virtual environments', size=12, weight='bold')
    _titled_box(ax, 0.4, 3.5, 2.4, 1.2, 'Project A', 'needs\nscikit-learn 1.9.1', center=True)
    _titled_box(ax, 3.2, 3.5, 2.4, 1.2, 'Project B', 'needs\nscikit-learn 0.24.2', center=True)
    _titled_box(ax, 0.4, 0.5, 5.2, 1.6, 'One global Python (one site-packages)',
                'only ONE scikit-learn version can be installed\n→ one project breaks',
                colors=(RED_T, RED), center=True)
    _arrow(ax, (1.6, 3.5), (2.2, 2.1))
    _arrow(ax, (4.4, 3.5), (3.8, 2.1))

    # --- with ---
    _text(ax, 9.0, 5.25, 'With virtual environments', size=12, weight='bold')
    _box(ax, 6.4, 0.2, 5.2, 4.7, fc=SURFACE, ec=AXIS, ls='--')
    _text(ax, 9.0, 0.45, 'one computer / server', size=9, color=INK2)
    for x, name, py, skl in [(6.7, 'Project A', '3.12', '1.9.1'),
                             (9.1, 'Project B', '3.8', '0.24.2')]:
        _titled_box(ax, x, 3.4, 2.2, 1.2, name, 'churn service' if py == '3.12' else 'old model',
                    center=True)
        _titled_box(ax, x, 0.8, 2.2, 1.9, 'own env',
                    f'Python {py}\nscikit-learn {skl}', colors=LAYER['env'], center=True)
        _arrow(ax, (x + 1.1, 3.4), (x + 1.1, 2.7))

    _finish(fig, 'venv_isolation', save, show)


def draw_vm_vs_containers(save=False, show=True):
    """Virtual machines virtualise hardware; containers share the host kernel."""
    fig, ax = _canvas(12, 6.2)

    def base(x0, middle, middle_colors):
        for y, label, colors in [(0.3, 'Hardware (CPU, RAM, disk)', (GRAY_T, AXIS)),
                                 (1.0, 'Host operating system + kernel', (GRAY_T, AXIS)),
                                 (1.7, middle, middle_colors)]:
            _titled_box(ax, x0, y, 5.2, 0.6, label, colors=colors, center=True, title_size=10)

    # virtual machines
    _text(ax, 3.0, 5.85, 'Virtual machines', size=12.5, weight='bold')
    _text(ax, 3.0, 5.4, 'each VM carries a full guest OS → GBs, minutes to boot',
          size=9.5, color=INK2)
    base(0.4, 'Hypervisor (VirtualBox, VMware, Hyper-V)', LAYER['cloud'])
    for i, app in enumerate('ABC'):
        x = 0.4 + i * 1.77
        _titled_box(ax, x, 2.4, 1.66, 1.0, 'Guest OS', 'own kernel', colors=(RED_T, RED),
                    center=True, title_size=10)
        _titled_box(ax, x, 3.5, 1.66, 0.6, 'libs', center=True, title_size=10,
                    colors=LAYER['env'])
        _titled_box(ax, x, 4.2, 1.66, 0.6, f'App {app}', center=True, title_size=10,
                    colors=LAYER['web'])

    # containers
    _text(ax, 9.0, 5.85, 'Containers (Docker)', size=12.5, weight='bold')
    _text(ax, 9.0, 5.4, 'containers share the host kernel → MBs, seconds to start',
          size=9.5, color=INK2)
    base(6.4, 'Docker Engine', LAYER['docker'])
    for i, app in enumerate('ABC'):
        x = 6.4 + i * 1.77
        _titled_box(ax, x, 2.4, 1.66, 0.6, 'libs', center=True, title_size=10,
                    colors=LAYER['env'])
        _titled_box(ax, x, 3.1, 1.66, 0.6, f'App {app}', center=True, title_size=10,
                    colors=LAYER['web'])

    _finish(fig, 'vm_vs_containers', save, show)


def draw_image_vs_containers(save=False, show=True):
    """One image (template) -> many containers (running instances)."""
    fig, ax = _canvas(12, 4.4)

    _titled_box(ax, 0.3, 1.5, 2.3, 1.4, 'Dockerfile', 'the recipe', center=True)
    _arrow(ax, (2.6, 2.2), (3.9, 2.2))
    _text(ax, 3.25, 2.5, 'docker build', size=9, family=MONO, color=INK2)

    _titled_box(ax, 3.9, 1.2, 2.7, 2.0, 'Image', 'churn-service\nread-only template\n(like a class)',
                colors=LAYER['docker'], center=True)

    for i, y in enumerate([3.0, 1.7, 0.4]):
        _titled_box(ax, 8.7, y, 3.0, 1.0, f'Container {i + 1}', 'running instance (like an object)',
                    colors=LAYER['env'], center=True, title_size=10.5)
        _arrow(ax, (6.6, 2.2), (8.7, y + 0.5))
    _text(ax, 7.55, 3.35, 'docker run', size=9, family=MONO, color=INK2)

    _finish(fig, 'image_vs_containers', save, show)


def draw_port_mapping(save=False, show=True):
    """How -p 127.0.0.1:8080:9696 connects your machine to the container."""
    fig, ax = _canvas(12, 5.8)

    _box(ax, 0.3, 0.9, 11.4, 4.6, fc=GRAY_T, ec=INK2)
    _text(ax, 0.55, 5.18, 'Your computer (the host) — Windows / Linux / macOS', size=11.5,
          weight='bold', ha='left')

    _titled_box(ax, 0.8, 2.1, 2.8, 1.8, 'Client', 'predict_client.py\nor a browser',
                colors=(SURFACE, INK2), center=True)

    _box(ax, 6.8, 1.3, 4.5, 3.4, *LAYER['docker'])
    _text(ax, 7.4, 4.35, 'Docker container (its own network)', size=10.5, weight='bold',
          ha='left')
    _titled_box(ax, 8.0, 2.1, 3.0, 1.7, 'gunicorn', 'listens on 0.0.0.0:9696\nserves predict:app',
                colors=LAYER['web'], center=True, title_size=11)

    for (cx, label, colors) in [(5.4, '8080', (YELLOW_T, YELLOW)), (6.8, '9696', LAYER['web'])]:
        ax.add_patch(Circle((cx, 3.0), 0.38, facecolor=colors[0], edgecolor=colors[1],
                            linewidth=2, zorder=3))
        _text(ax, cx, 3.0, label, size=10, weight='bold', family=MONO)
    for cx, label in [(5.4, 'host port'), (6.8, 'container port')]:
        ax.text(cx, 2.3, label, fontsize=9, color=INK2, ha='center', va='center', zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=GRAY_T, edgecolor='none'))

    _arrow(ax, (3.6, 3.0), (5.02, 3.0))
    _text(ax, 4.3, 3.35, 'localhost:8080', size=9, family=MONO, color=INK2)
    _arrow(ax, (5.78, 3.0), (6.42, 3.0))
    _arrow(ax, (7.18, 3.0), (8.0, 3.0))

    _text(ax, 6.0, 0.45, 'docker run -p 127.0.0.1:8080:9696 churn-service', size=11,
          family=MONO, weight='bold')
    _text(ax, 6.0, 0.05, '[host IP] : [host port] : [container port]', size=9.5,
          family=MONO, color=INK2)

    _finish(fig, 'port_mapping', save, show)


def draw_elastic_beanstalk(save=False, show=True):
    """What Elastic Beanstalk creates for us behind one public URL."""
    fig, ax = _canvas(12, 6.0)

    for i, y in enumerate([4.4, 3.3, 2.2]):
        _titled_box(ax, 0.2, y, 1.9, 0.75, f'client {i + 1}', center=True, title_size=10)
        _arrow(ax, (2.1, y + 0.375), (3.4, 3.1))
    _titled_box(ax, 0.2, 0.5, 1.9, 1.0, 'You', 'eb create / eb deploy', center=True,
                title_size=10.5)

    _box(ax, 2.8, 0.2, 9.0, 5.6, fc=SURFACE, ec=VIOLET, ls='--')
    _text(ax, 3.05, 5.45, 'AWS  ·  Elastic Beanstalk environment "churn-serving-env"',
          size=11, weight='bold', ha='left')

    _titled_box(ax, 3.4, 2.3, 2.5, 1.6, 'Load balancer', 'public URL\n*.elasticbeanstalk.com',
                colors=LAYER['cloud'], center=True, title_size=10.5)
    _titled_box(ax, 3.4, 0.5, 2.5, 1.2, 'EB manager', 'builds the image,\nwatches health',
                center=True, title_size=10.5)
    _arrow(ax, (2.1, 1.0), (3.4, 1.1))

    _box(ax, 6.5, 0.5, 5.0, 4.5, fc=SURFACE, ec=BLUE, ls='--')
    _text(ax, 6.75, 4.7, 'Auto Scaling group', size=10, weight='bold', ha='left', color=INK2)
    for i, (y, ls) in enumerate([(3.35, '-'), (2.15, '-'), (0.8, '--')]):
        sub = 'Docker: churn-service' if ls == '-' else 'added automatically when traffic grows'
        _titled_box(ax, 7.0, y, 4.2, 0.95, f'EC2 server {i + 1}', sub, colors=LAYER['docker'],
                    ls=ls, center=True, title_size=10)
        _arrow(ax, (5.9, 3.1), (7.0, y + 0.475), ls=ls)

    _finish(fig, 'elastic_beanstalk', save, show)


def draw_project_pipeline(save=False, show=True):
    """The whole project as a left-to-right pipeline."""
    fig, ax = _canvas(12, 2.6)
    steps = [
        ('Notebook', 'experiment', 'client', 'recap'),
        ('train.py', 'model_C=1.0.bin', 'model', '5.2'),
        ('predict.py', 'Flask service', 'web', '5.3 – 5.4'),
        ('Pipfile', 'Python deps', 'env', '5.5'),
        ('Dockerfile', 'system deps', 'docker', '5.6'),
        ('AWS EB', 'the cloud', 'cloud', '5.7'),
    ]
    w, gap = 1.62, 0.36
    for i, (title, sub, layer, section) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        _titled_box(ax, x, 0.8, w, 1.4, title, sub, colors=LAYER[layer], center=True,
                    title_size=10.5)
        _text(ax, x + w / 2, 0.4, f'§ {section}', size=9, color=INK2)
        if i < len(steps) - 1:
            _arrow(ax, (x + w, 1.5), (x + w + gap, 1.5))

    _finish(fig, 'project_pipeline', save, show)


ALL_DIAGRAMS = [draw_deployment_stack, draw_request_response, draw_venv_isolation,
                draw_vm_vs_containers, draw_image_vs_containers, draw_port_mapping,
                draw_elastic_beanstalk, draw_project_pipeline]


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    set_plot_style()
    for draw in ALL_DIAGRAMS:
        draw(save=True, show=False)
        print('saved', draw.__name__)
