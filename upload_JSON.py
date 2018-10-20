import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 跳过了远程连接中选择‘是’的环节,
ssh.connect('119.23.243.57', 22, 'root', 'Fdan123@')
transport = paramiko.Transport(('119.23.243.57', 22))
transport.connect(username='root', password='Fdan123@')
stdin, stdout, stderr = ssh.exec_command('df')#ssh 协议栈命令
for line in stdout:#逐行打印回显
    print(line)
sftp = paramiko.SFTPClient.from_transport(transport)
# 将location.py 上传至服务器 /tmp/test.py,都是绝对路径，不是文件夹
sftp.put('result.json', 'result/result.json')
# 将remove_path 下载到本地 local_path，都是绝对路径，不是文件夹
# sftp.get('result/hello.txt', 'hi.txt')
transport.close()
