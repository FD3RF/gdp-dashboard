# nosleep POC

## 使用方式
1. 前台任务：`nosleep ${CMD}`，比如 `nosleep sleep 1000`；
2. 后台任务：`nohup nosleep ${CMD} &`，比如 `nohup ./nosleep sleep 10000 > task.log 2>&1 &`；

说明：后台任务需要从日志文件比如 task.log 查看程序的标准输出和错误输出；后台任务能保障关闭浏览器后任务仍旧运行；

## 思路

1. 用户通过“设置 --> 访问令牌 --> 新增访问令牌”（不需要勾选任何权限），获取访问令牌；
![获取访问令牌](./img/Clipboard_Screenshot_1749526204.png)
2. 设置访问令牌环境变量 `export CS_HEALTHZ_TOKEN=xxx`；
3. 通过上文使用方式执行任务，比如，假设训练任务的命令是 `python train.py`，则执行 `./nosleep python train.py`；