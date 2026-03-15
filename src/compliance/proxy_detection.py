from typing import Dict, List, Any

def detect_proxies(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    suspicious = []
    for record in data:
        flags = []
        if record.get('ip_address', '').startswith('10.'):
            flags.append('Internal IP')
        if 'proxy' in str(record.get('user_agent', '')).lower():
            flags.append('Proxy UA')
        if flags:
            suspicious.append({'record_id': record.get('id'), 'flags': flags})
    return suspicious
