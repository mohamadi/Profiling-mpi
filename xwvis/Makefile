CC=mpicc
CFLAGS=-c
LDFLAGS=-lm -lmpe -L/usr/X11R6/lib -lX11
EXEC=vnbody
SRCS=vnbody.c
OBJS=$(SRCS:.c=.o)
$(EXEC):$(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)
.c.o:
	$(CC) $(CFLAGS) $< -o $@
