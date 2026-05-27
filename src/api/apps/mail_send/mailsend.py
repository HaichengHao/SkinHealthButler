# @Time    : 2026/5/27 10:13
# @Author  : hero
# @File    : mailsend.py
from email.mime.multipart import MIMEMultipart

from configs.project_default_configs import EMAIL_PWD, EMAIL_SENDER
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.header import Header
from email.utils import formataddr


def send_simple_email():
    # 邮件配置
    smtp_server = "smtp.qq.com"
    smtp_port = 587
    sender = EMAIL_SENDER
    password = EMAIL_PWD  # 注意：不是邮箱密码
    receiver = "target@qq.com"

    # 构建邮件内容
    message = MIMEText("这是一封由Python自动发送的测试邮件", 'plain', 'utf-8')
    message['From'] = formataddr(("Python自动化助手", sender))
    message['To'] = formataddr(("管理员", receiver))
    message['Subject'] = Header("系统通知：每日报告", 'utf-8')

    try:
        # 连接SMTP服务器并发送
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"发送失败：{e}")
    finally:
        server.quit()


# # 调用函数
# send_simple_email()


def send_html_email():
    # 邮件配置（同上，略）
    smtp_server = "smtp.qq.com"
    smtp_port = 587
    sender = EMAIL_SENDER
    password = EMAIL_PWD  # 注意：不是邮箱密码
    receiver = "target@qq.com"
    # 创建混合格式的邮件容器
    message = MIMEMultipart('alternative')
    message['From'] = formataddr(("Python自动化助手", sender))
    message['To'] = formataddr(("管理员", receiver))
    message['Subject'] = Header("系统通知：每日报告", 'utf-8')

    # HTML内容
    html_content = """
    <html>
      <body>
        <h2 style="color: #2c3e50;">销售数据周报</h2>
        <p>尊敬的团队成员，以下是本周的销售数据概览：</p>
        <table border="1" style="border-collapse: collapse;">
          <tr style="background-color: #f2f2f2;">
            <th>日期</th><th>销售额</th><th>完成率</th>
          </tr>
          <tr>
            <td>周一</td><td>¥12,450</td><td>85%</td>
          </tr>
          <tr>
            <td>周二</td><td>¥15,320</td><td>92%</td>
          </tr>
        </table>
        <p><em>数据来源：销售管理系统</em></p>
      </body>
    </html>
    """

    # 添加HTML内容到邮件
    html_part = MIMEText(html_content, 'html', 'utf-8')
    message.attach(html_part)
    try:
        # 连接SMTP服务器并发送
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"发送失败：{e}")
    finally:
        server.quit()


def send_email_with_attachment(file_path):
    smtp_server = "smtp.qq.com"
    smtp_port = 587
    sender = EMAIL_SENDER
    password = EMAIL_PWD  # 注意：不是邮箱密码
    receiver = "target@qq.com"
    # 创建混合格式的邮件
    message = MIMEMultipart()
    message['From'] = formataddr(("Python自动化助手", sender))
    message['To'] = formataddr(("管理员", receiver))
    message['Subject'] = Header("系统通知：每日报告", 'utf-8')

    # 邮件正文
    body = "您好，请查收2023年度财务报表，详情请见附件。"
    message.attach(MIMEText(body, 'plain', 'utf-8'))

    # 添加附件
    with open(file_path, mode='rb') as f:
        attachment = MIMEApplication(f.read())
        attachment.add_header(
            'Content-Disposition',
            'attachment',
            filename='01.png'
        )
        message.attach(attachment)
    try:
        # 连接SMTP服务器并发送
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"发送失败：{e}")
    finally:
        server.quit()

if __name__ == '__main__':
    # send_html_email()
    send_email_with_attachment('../../../../imgs/01.png')