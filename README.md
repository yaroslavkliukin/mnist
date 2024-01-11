Решаю задачу классификации рукописных цифр на дасете MNIST

## Тренировка

Сервер для логирования трейна

    mlflow server --host 127.0.0.1 --port 8890

Порт может быть занят, тогда при запуске трейна поменяйте параметр на нужный

    python mnist/train.py --tracking_uri=http://localhost:8890

## Инференс

Запуск сервера

    python mnist/run_server.py --port=8891

Тест, обратите внимание, что порт может поменяться

    python mnist/server_test.py --servind_addr=http://127.0.0.1:8891
