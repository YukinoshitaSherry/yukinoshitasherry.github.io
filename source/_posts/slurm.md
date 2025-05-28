---
title: Slurm使用教程
date: 2025-03-10
categories: 
    - 学CS/SE
tags: 
    - Slurm
    - Linux
    - GPU
desc: Slurm是一个开源的集群管理和作业调度系统，适用于Linux集群。跑实验申请GPU会用到它。
---

<br>
2025.3,在Yale做remote intern的时候接触到，使用 McCleary 集群申请 GPU 资源及作业，需要 SLURM 提交 GPU 作业。
<br>

# Slurm 使用教程

## 一、Slurm简介
- Slurm是一个开源的集群管理和作业调度系统，适用于Linux集群。
- 具有容错性、高度可扩展性，无需修改内核即可运行。
- 主要功能包括资源分配、作业执行框架提供以及资源争用仲裁。

<br>

## 二、Slurm环境配置
### 1. 登录节点
- 通常通过SSH客户端连接到集群的登录节点，如`ssh username@cluster_node`。
  - 在学校或公司内部，`cluster_node`可能是特定的服务器地址，如`hpc.usst.edu.cn`。
  - 部分集群可能提供图形化界面的登录方式，具体根据集群管理员设置。

### 2. 设置环境变量
- 根据集群要求，可能需要加载特定的模块或设置环境变量。
  - 常见命令如`module load slurm`，加载Slurm相关模块。
  - 可能还需要设置作业输出目录等环境变量，如`export OUTPUT_DIR=/path/to/output`。

### 3. 检查集群状态
- 使用`sinfo`命令查看集群的分区、节点状态等信息。
  - 如`sinfo -p partition_name`查看特定分区的状态。
  - 可以通过`man sinfo`查看详细参数说明。

<br>

## 三、Slurm作业管理
### 1. 提交作业
- **使用sbatch命令**：将作业脚本提交到Slurm系统，如`sbatch job_script.sh`。
  - 可以在命令中指定一些参数，如`-p`指定分区，`--job-name`指定作业名等。
- **指定参数**：可以在命令中或脚本内使用`#SBATCH`指定作业名、分区、资源需求等参数。
  - 例如：
    ```bash
    #SBATCH --job-name=test_job
    #SBATCH -p normal
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    ```

### 2. 查看作业状态
- **使用squeue命令**：查看当前所有作业的状态，如运行中、待调度等。
  - 如`squeue -u username`查看特定用户的作业。
  - 可以通过`man squeue`查看详细参数说明。

### 3. 取消作业
- **使用scancel命令**：通过作业ID取消指定的作业，如`scancel job_id`。
  - 也可以通过`scancel -u username`取消特定用户的全部作业。

<br>

## 四、Slurm资源查看
### 1. 查看节点信息
- 使用`sinfo`命令查看节点的可用性、状态等。
  - 如`sinfo -N`查看所有节点的信息。
  - 可以通过`man sinfo`查看详细参数说明。

### 2. 查看作业资源使用情况
- 使用`sacct`命令查看历史作业的资源使用情况。
  - 如`sacct -u username`查看特定用户的作业资源使用情况。
  - 可以通过`man sacct`查看详细参数说明。

<br>

## 五、Slurm文件操作
### 1. 日志文件管理
- 作业的输出和错误信息可以指定到特定的文件，方便后续查看和分析。
  - 在作业脚本中使用`#SBATCH --output`和`#SBATCH --error`指定输出文件。
  - 例如：
    ```bash
    #SBATCH --output=output/%j.out
    #SBATCH --error=output/%j.err
    ```

### 2. 文件传输
- 如果需要在节点间传输文件，可以使用集群提供的文件系统或工具。
  - 部分集群可能提供高速文件传输工具，具体根据集群管理员设置。

<br>

## 六、Slurm作业脚本示例
### 1. 串行作业示例
```bash
#!/bin/bash
#SBATCH --job-name=serial_job
#SBATCH -p serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# 加载所需的模块
module load gcc

# 执行串行程序
./serial_program
```

### 2. MPI作业示例
```bash
#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# 加载MPI模块
module load mpi

# 执行MPI程序
mpirun ./mpi_program
```

### 3. GPU作业示例
```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# 加载GPU相关的模块
module load cuda

# 执行GPU程序
./gpu_program
```
<br>

## 七、注意事项
- **资源申请合理**：根据作业需求合理申请资源，避免浪费。
- **错误处理**：在作业脚本中添加错误处理机制，确保作业的健壮性。
- **遵守集群规定**：遵循所在集群的使用规则和限制。
- **定期检查作业状态**：及时发现和处理作业运行中的问题。