web: sh -lc 'export PYTHONPATH=$PWD/src && uvicorn gradenza_api.main:app --host 0.0.0.0 --port $PORT'
worker: sh -lc 'export PYTHONPATH=$PWD/src && python -m gradenza_api.worker'
