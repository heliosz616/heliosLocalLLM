import hashlib
import json
from auth import principal,session_key

def test_keys_can_be_revoked_without_restart(tmp_path):
    p=tmp_path/'users.json'
    record={'alice':{'enabled':True,'sha256':hashlib.sha256(b'alice-key').hexdigest()}}
    p.write_text(json.dumps(record))
    assert principal('alice-key','master',p)=='alice'
    assert principal('wrong','master',p) is None
    assert principal('master','master',p)=='owner'
    record['alice']['enabled']=False
    p.write_text(json.dumps(record))
    assert principal('alice-key','master',p) is None

def test_same_session_id_isolated_by_principal():
    assert session_key('alice','A')!=session_key('bob','A')
    assert session_key('alice','A')!=session_key('owner','A')
    assert session_key('owner','A')=='A'
    assert session_key('alice',None) is None

def test_malformed_registry_and_unicode_fail_closed(tmp_path):
    p=tmp_path/'users.json'; p.write_text('{')
    assert principal('wrong','master',p) is None
    assert principal('不合法','master',p) is None
