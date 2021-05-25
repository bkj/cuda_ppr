include Makefile.inc

all: ppr

ppr : ppr.cu
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o ppr ppr.cu $(SOURCE) $(ARCH) $(INC)

clean:
	rm -f ppr