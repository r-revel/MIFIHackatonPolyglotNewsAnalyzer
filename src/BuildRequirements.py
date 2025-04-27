import yaml
from pathlib import Path

# Относительные пути
base_dir = Path(__file__).resolve().parent.parent  # поднимаемся из src/ на уровень выше
env_file = base_dir / "environment.yml"
req_file = base_dir / "requirements.txt"

# Чтение environment.yml
with open(env_file, 'r') as f:
    env_data = yaml.safe_load(f)

requirements = []

for dep in env_data['dependencies']:
    if isinstance(dep, str):
        # conda-зависимость
        if '=' in dep:
            name, version, *_ = dep.split('=')
            if name.lower() != 'python':  # исключаем python
                requirements.append(f"{name}=={version}")
        else:
            requirements.append(dep)
    elif isinstance(dep, dict) and 'pip' in dep:
        # pip-зависимости
        for pip_dep in dep['pip']:
            requirements.append(pip_dep)

# Убираем дубликаты и сортируем
requirements = sorted(set(requirements))

# Сохраняем в файл
with open(req_file, 'w') as f:
    for req in requirements:
        f.write(req + '\n')

print(f"✅ requirements.txt успешно создан по пути: {req_file}")