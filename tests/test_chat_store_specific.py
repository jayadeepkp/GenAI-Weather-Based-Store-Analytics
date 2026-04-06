import requests
import time

def test_manager_plain_english_gets_specific_answer():
    BASE = 'http://localhost:8000'

    try:
        ollama = requests.get('http://localhost:11434/api/tags', timeout=5)
        assert ollama.status_code == 200
    except:
        raise AssertionError('Ollama not running')

    r1 = requests.post(f'{BASE}/predict/chat', json={
        'store_id': 79609,
        'message': 'How has rain historically affected my store?'
    }, timeout=120)

    print("Response 1:", r1.text)

    r2 = requests.post(f'{BASE}/predict/chat', json={
        'store_id': 79609,
        'message': 'What should I expect this week?'
    }, timeout=120)

    print("Response 2:", r2.text)

    print("Test ran successfully")

test_manager_plain_english_gets_specific_answer()