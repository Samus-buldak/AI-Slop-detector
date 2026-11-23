"""Windows-friendly single entry point for local Qwen + .env chat/vision model.

Nothing is sent to the remote API until the user explicitly starts training
or submits text/image to analyse. Secrets stay in .env, not in the ZIP.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.api.local_qwen import LOCAL_ENDPOINT, MODEL_FOLDER
from src.core.scoring import v2_feature_names
from src.segmentation.segments import RULES_VERSION
from src.segmentation.segmentation_llm import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
TRAINED = ROOT / MODEL_FOLDER / 'thresholds.json'


def ask(prompt, default=''):
    try:
        answer = input(prompt).strip()
    except EOFError:
        raise SystemExit(0)
    return answer or default


def load_config():
    values = {}
    env = ROOT / '.env'
    if not env.is_file():
        return values
    for raw in env.read_text(encoding='utf-8-sig').splitlines():
        raw = raw.strip()
        if raw and not raw.startswith('#') and '=' in raw:
            key, value = raw.split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    return values


def configured(values):
    key = values.get('API_KEY', '').strip()
    return (bool(key) and key.lower() not in
            ('sk-...', 'sk-xxx', 'your_api_key_here', 'your_qwen_api_key_here',
             'paste_your_qwen_api_key_here', 'changeme', 'paste_your_api_key_here')
            and bool(values.get('MODEL', '').strip())
            and bool(values.get('API_URL', '').strip()))


def configure_env():
    env = ROOT / '.env'
    if not env.exists():
        shutil.copyfile(ROOT / '.env.example', env)
        print('Создан .env из шаблона. Вписать нужно только API_KEY, если API_URL/MODEL подходят.')
    values = load_config()
    if configured(values):
        print('OK: .env найден, API_KEY, API_URL и MODEL указаны (сам ключ не выводится).')
        return True
    print('Открой .env в текстовом редакторе и заполни API_KEY, API_URL и MODEL.')
    if sys.platform == 'win32':
        try:
            subprocess.Popen(['notepad.exe', str(env)], cwd=ROOT)
        except OSError:
            pass
    elif sys.platform == 'darwin':
        try:
            subprocess.Popen(['open', '-t', str(env)], cwd=ROOT)
        except OSError:
            pass
    ask('После сохранения .env нажми Enter... ')
    values = load_config()
    if not configured(values):
        print('Не вижу заполненных API_KEY / API_URL / MODEL в .env. Проверь файл и запусти снова.')
        return False
    print('OK: настройки LLM прочитаны.')
    return True


def run_local(*args):
    cmd = [PYTHON, '-m', 'src.api.local_qwen', *args]
    print('\n>>>', ' '.join(x for x in cmd[2:] if x != '--llm-from-env'), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def is_new_model_ready():
    required = (TRAINED, TRAINED.parent / 'tree.pkl',
                TRAINED.parent / 'centroids.json', TRAINED.parent / 'test_metrics.json')
    if not all(path.is_file() for path in required):
        return False
    try:
        settings = json.loads(TRAINED.read_text(encoding='utf-8'))
        centroids = json.loads((TRAINED.parent / 'centroids.json').read_text(encoding='utf-8'))
        return (settings.get('schema_version') == 2 and
                not settings.get('use_tfidf') and
                settings.get('with_llm_trained') is True and
                settings.get('llm_segmentation') is True and
                settings.get('segmentation_version') == PROMPT_VERSION and
                settings.get('feature_names') == v2_feature_names(extra_segments=True) and
                centroids.get('backend') == 'api' and
                centroids.get('model') == 'qwen3-embedding-local')
    except (OSError, ValueError, TypeError):
        return False



def _normalise_endpoint(value):
    return str(value or '').strip().rstrip('/')


def matching_chat_args(settings):
    """Return args for exactly the chat provider used during training.

    None means the saved chat model/endpoint cannot be recreated safely.
    """
    model = str(settings.get('llm_model') or '').strip()
    endpoint = _normalise_endpoint(settings.get('llm_endpoint'))
    if not model or not endpoint:
        return None
    if endpoint == _normalise_endpoint(LOCAL_ENDPOINT):
        return ['--llm-model', model]
    values = load_config()
    if (configured(values) and values.get('MODEL', '').strip() == model and
            _normalise_endpoint(values.get('API_URL')) == endpoint):
        return ['--llm-from-env']
    return None


def main():
    print('\n=== AI Slop Detector: запуск ===')
    print('1 — анализ с сохранёнными весами')
    print('2 — обучить модель со всеми 21 зонами и открыть анализ')
    print('0 — выход')
    choice = ask('Выбери: ', '1')
    if choice == '0':
        return 0
    if choice not in ('1', '2'):
        print('Нет такого пункта.')
        return 2
    train_new = choice == '2'
    if train_new and not configure_env():
        return 2
    print('\nПроверяю локальный Qwen (LM Studio, порт 1234)...')
    if run_local('check'):
        print('Запусти Local Server в LM Studio на порту 1234 и загрузи Qwen3 Embedding 0.6B')
        print('с идентификатором qwen3-embedding-local. Подробности в START_HERE_RU.txt.')
        return 3
    if train_new:
        print('Обучение заменит веса в artifacts/model_hybrid_qwen/.')
        print('Тексты будут отправлены chat-провайдеру из .env. Возможен расход квоты API.')
        if ask('Начать обучение? [д/Н]: ', 'н').lower() not in ('д', 'y', 'yes', 'да'):
            return 0
        if run_local('train', '--llm-from-env', '--with-llm', '--llm-segmentation'):
            return 4
        if not is_new_model_ready():
            print('Обучение не сохранило модель с 21 зоной. Проверь вывод выше.')
            return 5
    if not all((TRAINED.parent / name).is_file()
               for name in ('thresholds.json', 'tree.pkl', 'centroids.json')):
        print('Веса не найдены. Перезапусти меню и выбери обучение.')
        return 5
    settings = json.loads(TRAINED.read_text(encoding='utf-8'))
    extra_zones = settings.get('feature_names') == v2_feature_names(extra_segments=True)
    print('Модель использует', 'расширенные 21 аналитическую зону.' if extra_zones else '4 структурные зоны. Для расширенной схемы выбери обучение при запуске.')
    expected_version = PROMPT_VERSION if settings.get('llm_segmentation') else RULES_VERSION
    if settings.get('segmentation_version') != expected_version:
        print('[!] Сохранённые веса обучены с другой версией сегментации. Анализ будет UNCERTAIN до переобучения.')
    provider_args = []
    if settings.get('llm_segmentation'):
        if _normalise_endpoint(settings.get('llm_endpoint')) != _normalise_endpoint(LOCAL_ENDPOINT):
            if not configure_env():
                return 2
        provider_args = matching_chat_args(settings)
        if provider_args is None:
            print('[!] Не совпадает chat-модель или endpoint с теми, на которых обучалась сегментация.')
            print('Восстанови исходные API_URL/MODEL в .env или загрузи прежнюю chat-модель в LM Studio.')
            return 2
    elif settings.get('with_llm_trained'):
        matching_args = matching_chat_args(settings)
        if matching_args is None:
            print('[!] Chat-модель/endpoint отличаются от обучающих; будет использована независимая ветка no_llm.')
        elif ask('Включить chat-модель, совпадающую с обучением? [д/Н]: ', 'н').lower() in ('д', 'y', 'yes', 'да'):
            provider_args = matching_args
    elif configured(load_config()):
        print('В .env настроен chat-провайдер. Он может использоваться для пояснений/vision, но не меняет классификацию no_llm.')
        if ask('Включить chat/vision из .env? [д/Н]: ', 'н').lower() in ('д', 'y', 'yes', 'да'):
            provider_args = ['--llm-from-env']
    print('\nАнализ готов. Для выхода выбери 0.')
    while True:
        print('\n1 — проверить текст/SMS')
        print('2 — проверить текст + фотографию')
        print('3 — проверить EML-письмо (с автоматическим поиском фото)')
        print('0 — выход')
        choice = ask('Выбери: ', '0')
        if choice == '0':
            break
        if choice == '1':
            run_local('text', *provider_args)
        elif choice == '2':
            photo = ask('Путь к фото JPG/PNG/WEBP (можно перетащить файл в окно): ').strip('"')
            if not photo or not Path(photo).is_file():
                print('Файл не найден — выбери 1 для проверки только текста.')
            else:
                run_local('text', *provider_args, '--image', photo)
        elif choice == '3':
            eml = ask('Путь к .eml: ').strip('"')
            if not Path(eml).is_file():
                print('Файл не найден.')
            else:
                run_local('analyze', *provider_args, eml)
        else:
            print('Нет такого пункта.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nОстановлено пользователем.')
        raise SystemExit(130)
