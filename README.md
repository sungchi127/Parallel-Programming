# 平行程式設計作業總覽

本資料庫包含多個平行程式設計相關作業，涵蓋不同的平行計算模型與程式設計技術。每個作業皆針對特定的平行處理方法進行實作與測試。

## 📁 資料夾結構

- `Assignment_I_SIMD/` – SIMD 平行程式設計
- `Assignment_II_MultiThread/` – 多執行緒程式設計（Pthreads）
- `Assignment_III_OpenMP/` – OpenMP 平行程式設計
- `Assignment_IV_MPI/` – MPI 分散式程式設計
- `Assignment_V_CUDA/` – CUDA GPU 加速程式設計
- `Assignment_VI_OpenCL/` – OpenCL 跨平台平行程式設計

---

## 🧩 作業簡介

### 🟦 作業一：SIMD 程式設計
- **簡介**：使用 SIMD（單指令多資料）指令集進行資料層級的平行化處理。
- **使用技術**：SSE / AVX Intrinsics
- **目標**：透過向量化指令提升處理效率。

---

### 🟨 作業二：多執行緒程式設計
- **簡介**：透過 POSIX Threads（pthreads）實作多執行緒，展示任務層級的平行處理。
- **使用技術**：Pthreads
- **目標**：將工作分配至多個執行緒以加速處理。

---

### 🟧 作業三：OpenMP 程式設計
- **簡介**：利用 OpenMP 指令進行迴圈與程式區段的平行化。
- **使用技術**：OpenMP
- **目標**：簡化 CPU 上的平行處理實作流程。

---

### 🟥 作業四：MPI 程式設計
- **簡介**：使用 MPI 進行分散式記憶體環境下的平行運算。
- **使用技術**：Message Passing Interface (MPI)
- **目標**：透過程序間通訊實作大型分散式運算。

---

### 🟩 作業五：CUDA 程式設計
- **簡介**：運用 CUDA 在 NVIDIA GPU 上實作大量資料的平行處理。
- **使用技術**：CUDA C/C++
- **目標**：透過 GPU 加速運算，提高效能。

---

### 🟪 作業六：OpenCL 程式設計
- **簡介**：以 OpenCL 架構實作可跨平台的異質計算任務。
- **使用技術**：OpenCL
- **目標**：支援多種硬體平台（如 CPU、GPU、FPGA）進行平行運算。

