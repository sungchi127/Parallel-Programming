CXX := mpicxx

TARGET := matmul
HOST_FILE := hosts.txt
FLAG=-O3

all: $(TARGET)

$(TARGET): main.cc matmul.cc
	mpicxx $(FLAG) $^ -o $(TARGET) 


.PHONY: clean test run

run:
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_1
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_2
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_3
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_4
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_5
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data1_6
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_1
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_2
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_3
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_4
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_5
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < /home/.grade/HW4/data-set/data2_6

test:
	mpirun -np 4 --hostfile $(HOST_FILE) $(TARGET) < test

clean:
	rm -rf $(TARGET)
	