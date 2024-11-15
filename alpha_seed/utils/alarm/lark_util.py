#!/usr/bin/python
# coding=utf-8
import requests
import json


def get_token(app_id, app_secret):
    api_url = 'https://fsopen.bytedance.net/open-apis/auth/v3/tenant_access_token/internal/'
    data = {
        'app_id': app_id,
        'app_secret': app_secret,
    }

    resp = requests.post(api_url, json=data)
    return resp.json().get('tenant_access_token')


def send_message_to_employee(title, msg_content, email="yueyu@bytedance.com"):
    try:
        app_id = 'cli_a45ecf3bfb3a900e'
        app_secret = 'sCnh4PO0aqtrjq7eT04zkek8VGkQicSD'
        user_token = get_token(app_id, app_secret)
        url = "https://fsopen.bytedance.net/open-apis/im/v1/messages?receive_id_type=email"
        content = json.dumps({"zh_cn": {"title": title, "content": [[{"tag": "text", "text": msg_content}]]}})
        payload = json.dumps({"receive_id": email, "msg_type": "post", "receive_id_type": "email", "content": content})

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {user_token}'}
        response = requests.request("POST", url, headers=headers, data=payload)
    except Exception as e:
        print(f'Failed to send lark message: title={title}, msg_content={msg_content}, email={email}, exception={e}')


if __name__ == "__main__":
    print(send_message_to_employee("标题在这", "这是一条测试消息", "yueyu@bytedance.com"))
