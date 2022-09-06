## 测试DJ

### 1. 环境
测试过airsim==1.7.0

预编译UE4环境，测试过的Blocks[下载地址](https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/Blocks.zip)

使用本项目中的`settings.json`，放置于`C:\Users\Administrator\Documents\AirSim\`下或可执行文件同目录下

坐标系：
```
       / x
      /
     /
    /
   /
  ┌──────────►
  │          y
  │
  │
  │
  │
  ▼ z
```

### 2. 运行
```
# 1. 启动AirSim，如E:\AirsimEnv\Blocks\WindowsNoEditor\下
.\Blocks.exe -ResX=640 -ResY=480 -windowed
# 2. 运行rotation.py
python rotation.py
```

