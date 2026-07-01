import json
import time
import urllib.request

base = 'http://127.0.0.1:8010'
endpoints = [
    ('forecast', '/api/forecast?project_id=1'),
    ('monte_carlo', '/api/monte-carlo?project_id=1&simulations=1000'),
    ('similarity', '/api/similar-projects?project_id=1'),
    ('portfolio', '/api/portfolio'),
    ('drift', '/api/drift'),
]
results = []
for name, path in endpoints:
    start = time.perf_counter()
    with urllib.request.urlopen(base + path, timeout=30) as response:
        json.load(response)
    results.append({'endpoint': name, 'latency_ms': round((time.perf_counter() - start) * 1000, 1)})

post_endpoints = [
    ('assistant', '/api/assistant', {'project_id': 1, 'question': 'Why is this project risky?'}),
    ('scenario_lab', '/api/scenario-lab', {'project_id': 1, 'scenarios': [{'name': 'A', 'overrides': {'team_size': 12}}]}),
]
for name, path, payload in post_endpoints:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        json.load(response)
    results.append({'endpoint': name, 'latency_ms': round((time.perf_counter() - start) * 1000, 1)})

payload = {'benchmarks': results, 'server': base}
print(json.dumps(payload, indent=2))
with open('data/performance_benchmarks.json', 'w', encoding='utf-8') as file:
    json.dump(payload, file, indent=2)
