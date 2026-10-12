import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app import create_app
from src.ai import AIUnavailable, answer_question, document_passages

class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.config={'TESTING':True,'SECRET_KEY':'test-only','DATABASE':str(Path(self.tmp.name)/'test.db'),'ANSWER_FUNCTION':lambda question,history,doc,mode:'Test answer '+str(len(history))+' '+(doc['name'] if doc else '')}
        self.app=create_app(self.config)
        self.client=self.app.test_client()
    def tearDown(self): self.tmp.cleanup()
    def post(self,path,client=None,**kwargs):
        c=client or self.client
        token=c.get('/api/session').json['csrf_token']
        return c.post(path,headers={'X-CSRF-Token':token},**kwargs)
    def test_pages(self):
        for path in ['/','/landing','/chat','/lawyer','/lawyer_portal','/appointments','/mycases','/profileset','/cv','/form','/quiz','/case_study']:
            with self.subTest(path=path): self.assertEqual(self.client.get(path).status_code,200)
    def test_csrf(self):
        self.assertEqual(self.client.post('/get_response',data={'msg':'hello'}).status_code,403)
        token=self.client.get('/api/session').json['csrf_token']
        r=self.client.post('/get_response',data={'msg':'hello'},headers={'X-CSRF-Token':token,'Origin':'https://evil.example'})
        self.assertEqual(r.status_code,403)
    def test_chat_persistence_and_isolation(self):
        self.assertEqual(self.post('/get_response',data={'msg':'question'}).status_code,200)
        self.assertIn('Test answer 1',self.post('/get_response',data={'msg':'followup'}).text)
        self.assertEqual(len(self.client.get('/get_memory').json['history']),2)
        self.assertEqual(self.app.test_client().get('/get_memory').json['history'],[])
        self.assertEqual(self.post('/clear_memory').status_code,200)
        self.assertEqual(self.client.get('/get_memory').json['history'],[])
    def test_upload_and_followup(self):
        r=self.post('/upload_and_analyze',data={'file':(io.BytesIO(b'Example legal text'),'example.txt'),'msg':''})
        self.assertEqual(r.status_code,200)
        self.assertIn('example.txt',self.post('/get_response',data={'msg':'explain'}).text)
        self.post('/clear_file_memory')
        self.assertNotIn('example.txt',self.post('/get_response',data={'msg':'hello'}).text)
    def test_bad_uploads(self):
        for name,raw in [('x.exe',b'data'),('x.pdf',b'not a pdf'),('x.txt',b''),('x.txt',b'\xff')]:
            with self.subTest(name=name):self.assertEqual(self.post('/upload_and_analyze',data={'file':(io.BytesIO(raw),name)}).status_code,400)
        self.app.config['MAX_CONTENT_LENGTH']=100
        self.assertEqual(self.post('/upload_and_analyze',data={'file':(io.BytesIO(b'a'*1000),'x.txt')}).status_code,413)
    def test_validation(self):
        self.assertEqual(self.post('/get_response',data={}).status_code,400)
        self.assertEqual(self.post('/get_response',data={'msg':'x'*10001}).status_code,400)
        self.assertEqual(self.post('/set_memory_type',data={'k':'bad'}).status_code,400)
        self.assertEqual(self.post('/set_memory_type',data={'k':'-1'}).status_code,400)
        self.assertEqual(self.post('/set_memory_type',data={'k':'2','memory_type':'window'}).status_code,200)
        for i in range(4):self.post('/get_response',data={'msg':str(i)})
        self.assertEqual(len(self.client.get('/get_memory').json['history']),2)
    def test_records_and_authorization(self):
        r=self.post('/api/records/appointments',json={'title':'Test','date':'2099-10-12','time':'10:00'})
        self.assertEqual(r.status_code,201)
        rid=r.json['id']
        self.assertEqual(len(self.client.get('/api/records/appointments').json['records']),1)
        other=self.app.test_client();token=other.get('/api/session').json['csrf_token']
        self.assertEqual(other.get('/api/records/appointments').json['records'],[])
        self.assertEqual(other.delete('/api/records/appointments/'+rid,headers={'X-CSRF-Token':token}).status_code,404)
        self.assertEqual(self.post('/api/records/appointments',json={'title':'Old','date':'2000-01-01','time':'10:00'}).status_code,400)
        self.assertEqual(self.post('/api/records/cases',json={'title':'Case','description':'<script>alert(1)</script>'}).status_code,201)
        token=self.client.get('/api/session').json['csrf_token']
        self.assertEqual(self.client.delete('/api/records/appointments/'+rid,headers={'X-CSRF-Token':token}).status_code,200)
    def test_accounts_and_restart(self):
        self.post('/api/records/cases',json={'title':'Persistent'})
        cred={'email':'test@example.com','password':'long-test-password'}
        self.assertEqual(self.post('/api/auth/register',json=cred).status_code,200)
        self.assertEqual(len(self.client.get('/api/records/cases').json['records']),1)
        self.post('/api/auth/logout')
        self.assertEqual(self.client.get('/api/records/cases').json['records'],[])
        new=create_app(self.config).test_client()
        self.assertEqual(self.post('/api/auth/login',client=new,json=cred).status_code,200)
        self.assertEqual(len(new.get('/api/records/cases').json['records']),1)
        self.assertEqual(self.post('/api/auth/login',json={**cred,'password':'bad-password'}).status_code,401)
        self.assertEqual(self.post('/api/auth/register',json=cred).status_code,409)
    def test_provider_errors(self):
        def fail(*args):raise AIUnavailable('Not configured')
        self.app.config['ANSWER_FUNCTION']=fail
        self.assertEqual(self.post('/get_response',data={'msg':'hello'}).status_code,503)
        self.assertEqual(self.client.get('/get_memory').json['history'],[])
        with patch.dict(os.environ,{'GROQ_API_KEY':''}):
            with self.assertRaises(AIUnavailable):answer_question('hello',[])
    def test_ai_payload_and_sources(self):
        class FakeResponse:
            def __enter__(self):return io.StringIO(json.dumps({'choices':[{'message':{'content':'Answer [1]'}}]}))
            def __exit__(self,*args):pass
        with patch.dict(os.environ,{'GROQ_API_KEY':'test-key','GROQ_MODEL':'test-model'}), patch('src.ai.urlopen',return_value=FakeResponse()) as call:
            answer=answer_question('Summarize',[],{'name':'test.txt','text':'Test document.'})
            self.assertIn('test.txt',answer)
            payload=json.loads(call.call_args.args[0].data)
            self.assertEqual(payload['messages'][0]['role'],'system')
            self.assertEqual(payload['model'],'test-model')
    def test_passage_bounds(self):
        docs=document_passages({'name':'long','text':'abc '*60000},'abc')
        self.assertLessEqual(sum(len(t) for _,t in docs),18000)

if __name__=='__main__':unittest.main()
