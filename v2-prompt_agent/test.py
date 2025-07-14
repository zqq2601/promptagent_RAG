from openai import OpenAI
import torch, torchvision

# 设置 API 客户端
client = OpenAI(
    api_key="sk-6iitQXWCbLHOXDgt80794c9339B546719c4eFcB675378b61",
    base_url="https://twohornedcarp.com/v1"
)

def test_api_connection():
    try:
        response = client.chat.completions.create(
            model="qwen2.5-32b-instruct",
            messages=[
                {"role": "user", "content": "你好，请简要介绍一下你自己。"}
            ]
        )
        print("API 调用成功，返回内容如下：")
        print(response.choices[0].message.content)
    except Exception as e:
        print("API 调用失败，错误信息如下：")
        print(e)

if __name__ == "__main__":
    #test_api_connection()
    print(torch.__version__)           # 应显示 1.10.1+cu113
    print(torch.version.cuda)          # 应显示 11.3
    print(torch.cuda.is_available())   # 应返回 True
