# Load-Flow-Analysis-Optimization
This project performs load flow analysis and optimization on the IEEE 33-bus radial distribution system using Python. Two load flow methods, Direct Load Flow (DLF) and Backward-Forward Sweep (BFS), are implemented and compared.

# ⚡ Load Flow Analysis & Optimization of IEEE 33-Bus Radial Distribution System

## 📌 Overview

This project focuses on performing **load flow analysis** and **optimization** of the IEEE 33-bus radial distribution system using Python.

The system is analyzed using two efficient load flow techniques:

* Direct Load Flow (DLF) using BIBC/BCBV matrices
* Backward-Forward Sweep (BFS) method

Further, **Particle Swarm Optimization (PSO)** is applied to:

* Reduce power losses
* Improve voltage profile

Optimization is performed using:

* Distributed Generation (DG) placement
* Capacitor placement

---

## 🎯 Objectives

* Perform accurate load flow analysis on a radial system
* Compare DLF and BFS methods
* Analyze voltage profile and power losses
* Apply optimization techniques (PSO)
* Improve system efficiency and stability

---

## 🧠 Key Concepts Used

* Per Unit System
* Radial Distribution Networks
* BIBC & BCBV Matrices
* Direct Load Flow (DLF)
* Backward-Forward Sweep (BFS)
* Particle Swarm Optimization (PSO)

---

## 🏗️ System Details

* Test System: IEEE 33-Bus Radial Distribution System
* Base Voltage: 12.66 kV
* Base Power: 100 MVA
* Type: Radial Network

---

## ⚙️ Project Workflow

```
Input Data (CSV)
        ↓
Per Unit Conversion
        ↓
Load Flow Analysis (DLF + BFS)
        ↓
Voltage & Loss Analysis
        ↓
Optimization (PSO)
     ↙            ↘
DG Placement   Capacitor Placement
```

---

## 📊 Results Summary

### 🔹 Base Case

* Total Real Power Loss: **202.52 kW**
* Minimum Voltage: **0.913 p.u. (Bus 18)**
* Buses below 0.95 p.u.: **21 buses**

---

### 🔹 After DG Placement

* Loss Reduction: **~88%**
* Minimum Voltage Improved: **0.985 p.u.**
* All buses within acceptable limits

---

### 🔹 After Capacitor Placement

* Loss Reduction: **~27%**
* Improved voltage profile
* All buses above 0.95 p.u.

---

## 📈 Output Visualizations

The project generates multiple outputs:

* Voltage profile plots
* Power loss distribution graphs
* PSO convergence plots
* Branch current and loss analysis
* Comparison graphs (Before vs After optimization)

---

## 📁 Project Structure

```
├── src/
│   ├── main.py
│   ├── load_flow.py
│   ├── load_data.py
│   ├── losses.py
│   ├── dg_placement.py
│   ├── capacitor_placement.py
│   ├── build_matrices.py
│
├── data/
│   ├── branch_data.csv
│   ├── bus_data.csv
│
├── results/
│   ├── voltage plots
│   ├── loss plots
│   ├── optimization outputs
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Requirements

```bash
pip install numpy pandas matplotlib
```

### 3. Run Project

```bash
python main.py
```

---

## 🧪 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

---

## 📌 Key Observations

* DG placement provides **maximum improvement** in system performance
* Capacitor placement provides **moderate improvement**
* PSO effectively finds optimal locations and sizes

---

🔮 Future Scope

* Integration with Machine Learning models
* Real-time smart grid implementation
* Renewable energy integration
* Multi-objective optimization

---

👨‍💻 Author

**Dipanshu Tripathi**
Electrical Engineering

---

⭐ If you like this project

Give it a ⭐ on GitHub!
