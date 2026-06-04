from dotenv import load_dotenv
import os

print("当前目录:", os.getcwd())
print(".env存在?", os.path.exists(".env"))
load_dotenv()
psw = os.getenv("MYSQL_PASSWORD")
print("MYSQL_PASSWORD:", psw)