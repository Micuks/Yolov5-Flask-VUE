1,4c1
< # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
< """
< General utils
< """
---
> # General utils
6d2
< import contextlib
8,9d3
< import logging
< import math
14,15c8
< import shutil
< import signal
---
> import subprocess
17,19d9
< import urllib
< from itertools import repeat
< from multiprocessing.pool import ThreadPool
21,22d10
< from subprocess import check_output
< from zipfile import ZipFile
24a13
> import math
26,27d14
< import pandas as pd
< import pkg_resources as pkg
32,33c19,21
< from utils.downloads import gsutil_getsize
< from utils.metrics import box_iou, fitness
---
> from utils.google_utils import gsutil_getsize
> from utils.metrics import fitness
> from utils.torch_utils import init_torch_seeds
36,42d23
< FILE = Path(__file__).resolve()
< ROOT = FILE.parents[1]  # YOLOv5 root directory
< DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
< NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
< VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
< FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf
< 
44,164c25,28
< np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
< pd.options.display.max_columns = 10
< cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
< os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
< 
< 
< def is_kaggle():
<     # Is environment a Kaggle Notebook?
<     try:
<         assert os.environ.get('PWD') == '/kaggle/working'
<         assert os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'
<         return True
<     except AssertionError:
<         return False
< 
< 
< def is_writeable(dir, test=False):
<     # Return True if directory has write permissions, test opening a file with write permissions if test=True
<     if test:  # method 1
<         file = Path(dir) / 'tmp.txt'
<         try:
<             with open(file, 'w'):  # open file with write permissions
<                 pass
<             file.unlink()  # remove file
<             return True
<         except OSError:
<             return False
<     else:  # method 2
<         return os.access(dir, os.R_OK)  # possible issues on Windows
< 
< 
< def set_logging(name=None, verbose=VERBOSE):
<     # Sets level and returns logger
<     if is_kaggle():
<         for h in logging.root.handlers:
<             logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
<     rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
<     logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
<     return logging.getLogger(name)
< 
< 
< LOGGER = set_logging('yolov5')  # define globally (used in train.py, val.py, detect.py, etc.)
< 
< 
< def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
<     # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
<     env = os.getenv(env_var)
<     if env:
<         path = Path(env)  # use environment variable
<     else:
<         cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
<         path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
<         path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
<     path.mkdir(exist_ok=True)  # make if required
<     return path
< 
< 
< CONFIG_DIR = user_config_dir()  # Ultralytics settings dir
< 
< 
< class Profile(contextlib.ContextDecorator):
<     # Usage: @Profile() decorator or 'with Profile():' context manager
<     def __enter__(self):
<         self.start = time.time()
< 
<     def __exit__(self, type, value, traceback):
<         print(f'Profile results: {time.time() - self.start:.5f}s')
< 
< 
< class Timeout(contextlib.ContextDecorator):
<     # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
<     def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
<         self.seconds = int(seconds)
<         self.timeout_message = timeout_msg
<         self.suppress = bool(suppress_timeout_errors)
< 
<     def _timeout_handler(self, signum, frame):
<         raise TimeoutError(self.timeout_message)
< 
<     def __enter__(self):
<         signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
<         signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised
< 
<     def __exit__(self, exc_type, exc_val, exc_tb):
<         signal.alarm(0)  # Cancel SIGALRM if it's scheduled
<         if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
<             return True
< 
< 
< class WorkingDirectory(contextlib.ContextDecorator):
<     # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
<     def __init__(self, new_dir):
<         self.dir = new_dir  # new dir
<         self.cwd = Path.cwd().resolve()  # current dir
< 
<     def __enter__(self):
<         os.chdir(self.dir)
< 
<     def __exit__(self, exc_type, exc_val, exc_tb):
<         os.chdir(self.cwd)
< 
< 
< def try_except(func):
<     # try-except function. Usage: @try_except decorator
<     def handler(*args, **kwargs):
<         try:
<             func(*args, **kwargs)
<         except Exception as e:
<             print(e)
< 
<     return handler
< 
< 
< def methods(instance):
<     # Get class/instance methods
<     return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]
< 
< 
< def print_args(name, opt):
<     # Print argparser arguments
<     LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
---
> # format short g, %precision=5
> np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
> # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
> cv2.setNumThreads(0)
168,170d31
<     # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
<     # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
<     import torch.backends.cudnn as cudnn
173,179c34
<     torch.manual_seed(seed)
<     cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
< 
< 
< def intersect_dicts(da, db, exclude=()):
<     # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
<     return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
---
>     init_torch_seeds(seed)
188,245d42
< def is_docker():
<     # Is environment a Docker container?
<     return Path('/workspace').exists()  # or Path('/.dockerenv').exists()
< 
< 
< def is_colab():
<     # Is environment a Google Colab instance?
<     try:
<         import google.colab
<         return True
<     except ImportError:
<         return False
< 
< 
< def is_pip():
<     # Is file in a pip package?
<     return 'site-packages' in Path(__file__).resolve().parts
< 
< 
< def is_ascii(s=''):
<     # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
<     s = str(s)  # convert list, tuple, None, etc. to str
<     return len(s.encode().decode('ascii', 'ignore')) == len(s)
< 
< 
< def is_chinese(s='人工智能'):
<     # Is string composed of any Chinese characters?
<     return True if re.search('[\u4e00-\u9fff]', str(s)) else False
< 
< 
< def emojis(str=''):
<     # Return platform-dependent emoji-safe version of string
<     return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
< 
< 
< def file_size(path):
<     # Return file/dir size (MB)
<     path = Path(path)
<     if path.is_file():
<         return path.stat().st_size / 1E6
<     elif path.is_dir():
<         return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
<     else:
<         return 0.0
< 
< 
< def check_online():
<     # Check internet connectivity
<     import socket
<     try:
<         socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
<         return True
<     except OSError:
<         return False
< 
< 
< @try_except
< @WorkingDirectory(ROOT)
247,263c44,49
<     # Recommend 'git pull' if code is out of date
<     msg = ', for updates see https://github.com/ultralytics/yolov5'
<     s = colorstr('github: ')  # string
<     assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
<     assert not is_docker(), s + 'skipping check (Docker image)' + msg
<     assert check_online(), s + 'skipping check (offline)' + msg
< 
<     cmd = 'git fetch && git config --get remote.origin.url'
<     url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # git fetch
<     branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
<     n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
<     if n > 0:
<         s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
<     else:
<         s += f'up to date with {url} ✅'
<     LOGGER.info(emojis(s))  # emoji-safe
< 
---
>     # Suggest 'git pull' if repo is out of date
>     if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
>         s = subprocess.check_output(
>             'if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
>         if 'Your branch is behind' in s:
>             print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')
265,310d50
< def check_python(minimum='3.6.2'):
<     # Check current python version vs. required python version
<     check_version(platform.python_version(), minimum, name='Python ', hard=True)
< 
< 
< def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
<     # Check version vs. required version
<     current, minimum = (pkg.parse_version(x) for x in (current, minimum))
<     result = (current == minimum) if pinned else (current >= minimum)  # bool
<     s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
<     if hard:
<         assert result, s  # assert min requirements met
<     if verbose and not result:
<         LOGGER.warning(s)
<     return result
< 
< 
< @try_except
< def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
<     # Check installed dependencies meet requirements (pass *.txt file or list of packages)
<     prefix = colorstr('red', 'bold', 'requirements:')
<     check_python()  # check python version
<     if isinstance(requirements, (str, Path)):  # requirements.txt file
<         file = Path(requirements)
<         assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
<         with file.open() as f:
<             requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
<     else:  # list or tuple of packages
<         requirements = [x for x in requirements if x not in exclude]
< 
<     n = 0  # number of packages updates
<     for r in requirements:
<         try:
<             pkg.require(r)
<         except Exception:  # DistributionNotFound or VersionConflict if requirements not met
<             s = f"{prefix} {r} not found and is required by YOLOv5"
<             if install:
<                 LOGGER.info(f"{s}, attempting auto-update...")
<                 try:
<                     assert check_online(), f"'pip install {r}' skipped (offline)"
<                     LOGGER.info(check_output(f"pip install '{r}'", shell=True).decode())
<                     n += 1
<                 except Exception as e:
<                     LOGGER.warning(f'{prefix} {e}')
<             else:
<                 LOGGER.info(f'{s}. Please install and rerun your command.')
312,326c52,57
<     if n:  # if packages updated
<         source = file.resolve() if 'file' in locals() else requirements
<         s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
<             f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
<         LOGGER.info(emojis(s))
< 
< 
< def check_img_size(imgsz, s=32, floor=0):
<     # Verify image size is a multiple of stride s in each dimension
<     if isinstance(imgsz, int):  # integer i.e. img_size=640
<         new_size = max(make_divisible(imgsz, int(s)), floor)
<     else:  # list i.e. img_size=[640, 480]
<         new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
<     if new_size != imgsz:
<         LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
---
> def check_img_size(img_size, s=32):
>     # Verify img_size is a multiple of stride s
>     new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
>     if new_size != img_size:
>         print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' %
>               (img_size, s, new_size))
330,365c61,63
< def check_imshow():
<     # Check if environment supports image displays
<     try:
<         assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
<         assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
<         cv2.imshow('test', np.zeros((1, 1, 3)))
<         cv2.waitKey(1)
<         cv2.destroyAllWindows()
<         cv2.waitKey(1)
<         return True
<     except Exception as e:
<         LOGGER.warning(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
<         return False
< 
< 
< def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
<     # Check file(s) for acceptable suffix
<     if file and suffix:
<         if isinstance(suffix, str):
<             suffix = [suffix]
<         for f in file if isinstance(file, (list, tuple)) else [file]:
<             s = Path(f).suffix.lower()  # file suffix
<             if len(s):
<                 assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"
< 
< 
< def check_yaml(file, suffix=('.yaml', '.yml')):
<     # Search/download YAML file (if necessary) and return path, checking suffix
<     return check_file(file, suffix)
< 
< 
< def check_file(file, suffix=''):
<     # Search/download file (if necessary) and return path
<     check_suffix(file, suffix)  # optional
<     file = str(file)  # convert to str()
<     if Path(file).is_file() or file == '':  # exists
---
> def check_file(file):
>     # Search for file if not found
>     if os.path.isfile(file) or file == '':
367,382c65,69
<     elif file.startswith(('http:/', 'https:/')):  # download
<         url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
<         file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
<         if Path(file).is_file():
<             LOGGER.info(f'Found {url} locally at {file}')  # file already exists
<         else:
<             LOGGER.info(f'Downloading {url} to {file}...')
<             torch.hub.download_url_to_file(url, file)
<             assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
<         return file
<     else:  # search
<         files = []
<         for d in 'data', 'models', 'utils':  # search directories
<             files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
<         assert len(files), f'File not found: {file}'  # assert file was found
<         assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
---
>     else:
>         files = glob.glob('./**/' + file, recursive=True)  # find file
>         assert len(files), 'File Not Found: %s' % file  # assert file was found
>         assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (
>             file, files)  # assert unique
386,425c73,78
< def check_font(font=FONT):
<     # Download font to CONFIG_DIR if necessary
<     font = Path(font)
<     if not font.exists() and not (CONFIG_DIR / font.name).exists():
<         url = "https://ultralytics.com/assets/" + font.name
<         LOGGER.info(f'Downloading {url} to {CONFIG_DIR / font.name}...')
<         torch.hub.download_url_to_file(url, str(font), progress=False)
< 
< 
< def check_dataset(data, autodownload=True):
<     # Download and/or unzip dataset if not found locally
<     # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip
< 
<     # Download (optional)
<     extract_dir = ''
<     if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
<         download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False, threads=1)
<         data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
<         extract_dir, autodownload = data.parent, False
< 
<     # Read yaml (optional)
<     if isinstance(data, (str, Path)):
<         with open(data, errors='ignore') as f:
<             data = yaml.safe_load(f)  # dictionary
< 
<     # Resolve paths
<     path = Path(extract_dir or data.get('path') or '')  # optional 'path' default to '.'
<     if not path.is_absolute():
<         path = (ROOT / path).resolve()
<     for k in 'train', 'val', 'test':
<         if data.get(k):  # prepend path
<             data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
< 
<     # Parse yaml
<     assert 'nc' in data, "Dataset 'nc' key missing."
<     if 'names' not in data:
<         data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
<     train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
<     if val:
<         val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
---
> def check_dataset(dict):
>     # Download dataset if not found locally
>     val, s = dict.get('val'), dict.get('download')
>     if val and len(val):
>         val = [Path(x).resolve()
>                for x in (val if isinstance(val, list) else [val])]  # val path
427,429c80,83
<             LOGGER.info('\nDataset not found, missing paths: %s' % [str(x) for x in val if not x.exists()])
<             if s and autodownload:  # download script
<                 root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
---
>             print('\nWARNING: Dataset not found, nonexistent paths: %s' %
>                   [str(x) for x in val if not x.exists()])
>             if s and len(s):  # download script
>                 print('Downloading %s ...' % s)
432d85
<                     LOGGER.info(f'Downloading {s} to {f}...')
434,439c87,89
<                     Path(root).mkdir(parents=True, exist_ok=True)  # create root
<                     ZipFile(f).extractall(path=root)  # unzip
<                     Path(f).unlink()  # remove zip
<                     r = None  # success
<                 elif s.startswith('bash '):  # bash script
<                     LOGGER.info(f'Running {s} ...')
---
>                     r = os.system('unzip -q %s -d ../ && rm %s' %
>                                   (f, f))  # unzip
>                 else:  # bash script
441,443c91,92
<                 else:  # python script
<                     r = exec(s, {'yaml': data})  # return None
<                 LOGGER.info(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
---
>                 print('Dataset autodownload %s\n' % ('success' if r ==
>                                                      0 else 'failure'))  # analyze return value
447,489d95
<     return data  # dictionary
< 
< 
< def url2file(url):
<     # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
<     url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
<     file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
<     return file
< 
< 
< def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
<     # Multi-threaded file download and unzip function, used in data.yaml for autodownload
<     def download_one(url, dir):
<         # Download 1 file
<         f = dir / Path(url).name  # filename
<         if Path(url).is_file():  # exists in current path
<             Path(url).rename(f)  # move to dir
<         elif not f.exists():
<             LOGGER.info(f'Downloading {url} to {f}...')
<             if curl:
<                 os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
<             else:
<                 torch.hub.download_url_to_file(url, f, progress=True)  # torch download
<         if unzip and f.suffix in ('.zip', '.gz'):
<             LOGGER.info(f'Unzipping {f}...')
<             if f.suffix == '.zip':
<                 ZipFile(f).extractall(path=dir)  # unzip
<             elif f.suffix == '.gz':
<                 os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
<             if delete:
<                 f.unlink()  # remove zip
< 
<     dir = Path(dir)
<     dir.mkdir(parents=True, exist_ok=True)  # make directory
<     if threads > 1:
<         pool = ThreadPool(threads)
<         pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
<         pool.close()
<         pool.join()
<     else:
<         for u in [url] if isinstance(url, (str, Path)) else url:
<             download_one(u, dir)
< 
492,494c98
<     # Returns nearest x divisible by divisor
<     if isinstance(divisor, torch.Tensor):
<         divisor = int(divisor.max())  # to int
---
>     # Returns x evenly divisible by divisor
498,532d101
< def clean_str(s):
<     # Cleans a string by replacing special characters with underscore _
<     return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
< 
< 
< def one_cycle(y1=0.0, y2=1.0, steps=100):
<     # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
<     return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
< 
< 
< def colorstr(*input):
<     # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
<     *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
<     colors = {'black': '\033[30m',  # basic colors
<               'red': '\033[31m',
<               'green': '\033[32m',
<               'yellow': '\033[33m',
<               'blue': '\033[34m',
<               'magenta': '\033[35m',
<               'cyan': '\033[36m',
<               'white': '\033[37m',
<               'bright_black': '\033[90m',  # bright colors
<               'bright_red': '\033[91m',
<               'bright_green': '\033[92m',
<               'bright_yellow': '\033[93m',
<               'bright_blue': '\033[94m',
<               'bright_magenta': '\033[95m',
<               'bright_cyan': '\033[96m',
<               'bright_white': '\033[97m',
<               'end': '\033[0m',  # misc
<               'bold': '\033[1m',
<               'underline': '\033[4m'}
<     return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
< 
< 
554c123,124
<     class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
---
>     class_counts = np.array(
>         [np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
592,647d161
< def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
<     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
<     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
<     y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
<     y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
<     y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
<     y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
<     return y
< 
< 
< def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
<     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
<     if clip:
<         clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
<     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
<     y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
<     y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
<     y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
<     y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
<     return y
< 
< 
< def xyn2xy(x, w=640, h=640, padw=0, padh=0):
<     # Convert normalized segments into pixel segments, shape (n,2)
<     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
<     y[:, 0] = w * x[:, 0] + padw  # top left x
<     y[:, 1] = h * x[:, 1] + padh  # top left y
<     return y
< 
< 
< def segment2box(segment, width=640, height=640):
<     # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
<     x, y = segment.T  # segment xy
<     inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
<     x, y, = x[inside], y[inside]
<     return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy
< 
< 
< def segments2boxes(segments):
<     # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
<     boxes = []
<     for s in segments:
<         x, y = s.T  # segment xy
<         boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
<     return xyxy2xywh(np.array(boxes))  # cls, xywh
< 
< 
< def resample_segments(segments, n=1000):
<     # Up-sample an (n,2) segment
<     for i, s in enumerate(segments):
<         x = np.linspace(0, len(s) - 1, n)
<         xp = np.arange(len(s))
<         segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
<     return segments
< 
< 
651,652c165,168
<         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
<         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
---
>         gain = min(img1_shape[0] / img0_shape[0],
>                    img1_shape[1] / img0_shape[1])  # gain  = old / new
>         pad = (img1_shape[1] - img0_shape[1] * gain) / \
>             2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
664c180
< def clip_coords(boxes, shape):
---
> def clip_coords(boxes, img_shape):
666,673c182,268
<     if isinstance(boxes, torch.Tensor):  # faster individually
<         boxes[:, 0].clamp_(0, shape[1])  # x1
<         boxes[:, 1].clamp_(0, shape[0])  # y1
<         boxes[:, 2].clamp_(0, shape[1])  # x2
<         boxes[:, 3].clamp_(0, shape[0])  # y2
<     else:  # np.array (faster grouped)
<         boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
<         boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
---
>     boxes[:, 0].clamp_(0, img_shape[1])  # x1
>     boxes[:, 1].clamp_(0, img_shape[0])  # y1
>     boxes[:, 2].clamp_(0, img_shape[1])  # x2
>     boxes[:, 3].clamp_(0, img_shape[0])  # y2
> 
> 
> def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
>     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
>     box2 = box2.T
> 
>     # Get the coordinates of bounding boxes
>     if x1y1x2y2:  # x1, y1, x2, y2 = box1
>         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
>         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
>     else:  # transform from xywh to xyxy
>         b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
>         b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
>         b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
>         b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
> 
>     # Intersection area
>     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
>             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
> 
>     # Union Area
>     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
>     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
>     union = w1 * h1 + w2 * h2 - inter + eps
> 
>     iou = inter / union
>     if GIoU or DIoU or CIoU:
>         # convex (smallest enclosing box) width
>         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
>         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
>         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
>             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
>             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
>                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
>             if DIoU:
>                 return iou - rho2 / c2  # DIoU
>             elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
>                 v = (4 / math.pi ** 2) * \
>                     torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
>                 with torch.no_grad():
>                     alpha = v / ((1 + eps) - iou + v)
>                 return iou - (rho2 / c2 + v * alpha)  # CIoU
>         else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
>             c_area = cw * ch + eps  # convex area
>             return iou - (c_area - union) / c_area  # GIoU
>     else:
>         return iou  # IoU
> 
> 
> def box_iou(box1, box2):
>     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
>     """
>     Return intersection-over-union (Jaccard index) of boxes.
>     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
>     Arguments:
>         box1 (Tensor[N, 4])
>         box2 (Tensor[M, 4])
>     Returns:
>         iou (Tensor[N, M]): the NxM matrix containing the pairwise
>             IoU values for every element in boxes1 and boxes2
>     """
> 
>     def box_area(box):
>         # box = 4xn
>         return (box[2] - box[0]) * (box[3] - box[1])
> 
>     area1 = box_area(box1.T)
>     area2 = box_area(box2.T)
> 
>     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
>     inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
>              torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
>     # iou = inter / (area1 + area2 - inter)
>     return inter / (area1[:, None] + area2 - inter)
> 
> 
> def wh_iou(wh1, wh2):
>     # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
>     wh1 = wh1[:, None]  # [N,1,2]
>     wh2 = wh2[None]  # [1,M,2]
>     inter = torch.min(wh1, wh2).prod(2)  # [N,M]
>     # iou = inter / (area1 + area2 - inter)
>     return inter / (wh1.prod(2) + wh2.prod(2) - inter)
676,678c271,272
< def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
<                         labels=(), max_det=300):
<     """Runs Non-Maximum Suppression (NMS) on inference results
---
> def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None, agnostic=False, labels=()):
>     """Performs Non-Maximum Suppression (NMS) on inference results
681c275
<          list of detections, on (n,6) tensor per image [xyxy, conf, cls]
---
>          detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
684c278
<     nc = prediction.shape[2] - 5  # number of classes
---
>     nc = prediction[0].shape[1] - 5  # number of classes
687,690d280
<     # Checks
<     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
<     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
< 
692,693c282,284
<     min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
<     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
---
>     # (pixels) minimum and maximum box width and height
>     min_wh, max_wh = 2, 4096
>     max_det = 300  # maximum number of detections per image
696c287
<     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
---
>     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
700c291
<     output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
---
>     output = [torch.zeros(0, 6)] * prediction.shape[0]
703c294
<         x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
---
>         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
708,710c299,301
<             lb = labels[xi]
<             v = torch.zeros((len(lb), nc + 5), device=x.device)
<             v[:, :4] = lb[:, 1:5]  # box
---
>             l = labels[xi]
>             v = torch.zeros((len(l), nc + 5), device=x.device)
>             v[:, :4] = l[:, 1:5]  # box
712c303
<             v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
---
>             v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
731c322,323
<             x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
---
>             x = torch.cat((box, conf, j.float()), 1)[
>                 conf.view(-1) > conf_thres]
734c326
<         if classes is not None:
---
>         if classes:
741c333
<         # Check shape
---
>         # If none remain process next image
743c335
<         if not n:  # no boxes
---
>         if not n:
745,746c337,339
<         elif n > max_nms:  # excess boxes
<             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
---
> 
>         # Sort by confidence
>         # x = x[x[:, 4].argsort(descending=True)]
750c343,344
<         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
---
>         # boxes (offset by class), scores
>         boxes, scores = x[:, :4] + c, x[:, 4]
758c352,353
<             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
---
>             x[i, :4] = torch.mm(weights, x[:, :4]).float(
>             ) / weights.sum(1, keepdim=True)  # merged boxes
764d358
<             LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
770c364,365
< def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
---
> # from utils.general import *; strip_optimizer()
> def strip_optimizer(f='weights/best.pt', s=''):
773,776c368,369
<     if x.get('ema'):
<         x['model'] = x['ema']  # replace model with ema
<     for k in 'optimizer', 'best_fitness', 'wandb_id', 'ema', 'updates':  # keys
<         x[k] = None
---
>     x['optimizer'] = None
>     x['training_results'] = None
783c376,377
<     LOGGER.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")
---
>     print('Optimizer stripped from %s,%s %.1fMB' %
>           (f, (' saved as %s,' % s) if s else '', mb))
786,793c380,386
< def print_mutation(results, hyp, save_dir, bucket):
<     evolve_csv = save_dir / 'evolve.csv'
<     evolve_yaml = save_dir / 'hyp_evolve.yaml'
<     keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
<             'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
<     keys = tuple(x.strip() for x in keys)
<     vals = results + tuple(hyp.values())
<     n = len(keys)
---
> def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
>     # Print mutation results to evolve.txt (for use with train.py --evolve)
>     a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
>     b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
>     # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
>     c = '%10.4g' * len(results) % results
>     print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
795d387
<     # Download (optional)
797,808c389,399
<         url = f'gs://{bucket}/evolve.csv'
<         if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
<             os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local
< 
<     # Log to evolve.csv
<     s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
<     with open(evolve_csv, 'a') as f:
<         f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')
< 
<     # Print to screen
<     LOGGER.info(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
<     LOGGER.info(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals) + '\n\n')
---
>         url = 'gs://%s/evolve.txt' % bucket
>         if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
>             # download evolve.txt if larger than local
>             os.system('gsutil cp %s .' % url)
> 
>     with open('evolve.txt', 'a') as f:  # append result
>         f.write(c + b + '\n')
>     x = np.unique(np.loadtxt('evolve.txt', ndmin=2),
>                   axis=0)  # load unique rows
>     x = x[np.argsort(-fitness(x))]  # sort
>     np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness
811,820c402,410
<     with open(evolve_yaml, 'w') as f:
<         data = pd.read_csv(evolve_csv)
<         data = data.rename(columns=lambda x: x.strip())  # strip keys
<         i = np.argmax(fitness(data.values[:, :7]))  #
<         f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
<                 f'# Best generation: {i}\n' +
<                 f'# Last generation: {len(data) - 1}\n' +
<                 '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
<                 '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
<         yaml.safe_dump(hyp, f, sort_keys=False)
---
>     for i, k in enumerate(hyp.keys()):
>         hyp[k] = float(x[0, i + 7])
>     with open(yaml_file, 'w') as f:
>         results = tuple(x[0, :7])
>         # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
>         c = '%10.4g' * len(results) % results
>         f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(
>             x) + c + '\n\n')
>         yaml.dump(hyp, f, sort_keys=False)
823c413,414
<         os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload
---
>         os.system('gsutil cp evolve.txt %s gs://%s' %
>                   (yaml_file, bucket))  # upload
827,828c418
<     # Apply a second stage classifier to YOLO outputs
<     # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
---
>     # applies a second stage classifier to yolo outputs
849c439
<                 # cv2.imwrite('example%i.jpg' % j, cutout)
---
>                 # cv2.imwrite('test%i.jpg' % j, cutout)
851,853c441,445
<                 im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
<                 im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
<                 im /= 255  # 0 - 255 to 0.0 - 1.0
---
>                 # BGR to RGB, to 3x416x416
>                 im = im[:, :, ::-1].transpose(2, 0, 1)
>                 im = np.ascontiguousarray(
>                     im, dtype=np.float32)  # uint8 to float32
>                 im /= 255.0  # 0 - 255 to 0.0 - 1.0
856,857c448,451
<             pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
<             x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections
---
>             pred_cls2 = model(torch.Tensor(ims).to(d.device)
>                               ).argmax(1)  # classifier prediction
>             # retain matching class detections
>             x[i] = x[i][pred_cls1 == pred_cls2]
862,863c456,457
< def increment_path(path, exist_ok=False, sep='', mkdir=False):
<     # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
---
> def increment_path(path, exist_ok=True, sep=''):
>     # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
865,866c459,461
<     if path.exists() and not exist_ok:
<         path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
---
>     if (path.exists() and exist_ok) or (not path.exists()):
>         return str(path)
>     else:
871,874c466,467
<         path = Path(f"{path}{sep}{n}{suffix}")  # increment path
<     if mkdir:
<         path.mkdir(parents=True, exist_ok=True)  # make directory
<     return path
---
>         return f"{path}{sep}{n}"  # update path
> 
875a469,501
> def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
>     shape = img.shape[:2]  # current shape [height, width]
>     if isinstance(new_shape, int):
>         new_shape = (new_shape, new_shape)
> 
>     # Scale ratio (new / old)
>     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
>     if not scaleup:  # only scale down, do not scale up (for better test mAP)
>         r = min(r, 1.0)
> 
>     # Compute padding
>     ratio = r, r  # width, height ratios
>     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
>     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
>         new_unpad[1]  # wh padding
>     if auto:  # minimum rectangle
>         dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
>     elif scaleFill:  # stretch
>         dw, dh = 0.0, 0.0
>         new_unpad = (new_shape[1], new_shape[0])
>         ratio = new_shape[1] / shape[1], new_shape[0] / \
>             shape[0]  # width, height ratios
> 
>     dw /= 2  # divide padding into 2 sides
>     dh /= 2
> 
>     if shape[::-1] != new_unpad:  # resize
>         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
>     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
>     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
>     img = cv2.copyMakeBorder(
>         img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
>     return img, ratio, (dw, dh)
877,878d502
< # Variables
< NCOLS = 0 if is_docker() else shutil.get_terminal_size().columns  # terminal window size for tqdm
