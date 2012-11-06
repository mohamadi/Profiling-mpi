/*
 	CPSC 521: Parallel Algorithms and Architectures
	Assignment 2: Performance Evaluation
	Author: Hamid Mohamadi, mohamadi@alumni.ubc.ca
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "mpi.h"
#include "mpe.h"
#include "mpe_graphics.h"

typedef struct{
	double x,y;
	double m;
} body;

typedef struct{
	double x, y;
} fvp; // data type for force and velocity

void frcInit(const int , body *, fvp *); /* computing internal forces */
void frcUpdt(const int , body *, body *, fvp *); /* updating the forces */
void posUpdt(const int , body *, fvp *, fvp *); /* updating positions */
void scatter(const char *, int, int, const int, body *);
void simulate(const int, int, int, const int, body *);
void gather(int, int, const int, body *);

int main(int argc, char *argv[]){    
	const int tmax=atoi(argv[1]); /* number of simulation rounds */
	const int gran=atoi(argv[2]); /* granularity */
	const char *pName=argv[3]; /* bodies file name */
	double time, time1, time2, time3; /* total, scatter, simulate, and gather running times */ 
	int rank, size; /* rank of process, number of processes */
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	body *bodies = (body *) malloc(gran*sizeof(body));

	time = MPI_Wtime();
	/****************************************************************
	1. Scatter:
	Reading the input from file and assigning each line to a process
	****************************************************************/
	time1 = MPI_Wtime();
	scatter(pName, rank, size, gran, bodies);
	time1 = MPI_Wtime() - time1;
	if(rank==0) printf("Scatter time is %.4f seconds\n", time1);	
	/****************************************************************
	2. Simulate:
	Passing the data of each process to other processes and simulate
	****************************************************************/
	time2 = MPI_Wtime();
	simulate(tmax, rank, size, gran, bodies);
	time2 = MPI_Wtime() - time2;
	if(rank==0) printf("Simulate time is %.4f seconds\n", time2);	
	/****************************************************************
	3. Gather:
	Writing the the last positions to "peval_out.txt" 
	****************************************************************/
	time3 = MPI_Wtime();
	gather(rank, size, gran, bodies);
	time3 = MPI_Wtime() - time3;
	if(rank==0) printf("Gather time is %.4f seconds\n", time3);	
	
	/* Finalizing */
	free(bodies);
	time = MPI_Wtime() - time;
	if(rank==0) printf("Computation time is %.4f seconds\n", time);	
	MPI_Finalize();
	return(0);
}

void frcInit(const int gran, body *bodies, fvp *f){
	int i,j;
	double r; 
	const double  G =  0.00667384;
	for(i=0; i< gran; i++)
		f[i].x=f[i].y=0;
	for(i=0; i<gran-1; i++){
		r=0; //f[i].x=0; f[i].y=0;
		for(j=i+1; j < gran; j++)
			if(j!=i){
				r = sqrt((bodies[j].x-bodies[i].x)*(bodies[j].x-bodies[i].x)+
					     (bodies[j].y-bodies[i].y)*(bodies[j].y-bodies[i].y));
				f[i].x += G*bodies[i].m*bodies[j].m*(bodies[j].x-bodies[i].x)/(r*r*r);
				f[i].y += G*bodies[i].m*bodies[j].m*(bodies[j].y-bodies[i].y)/(r*r*r);
				f[j].x += -f[i].x;
				f[j].y += -f[i].y;
			}
	}
}

void frcUpdt(const int gran, body *bodies, body *rcvBodies, fvp *f){
	int i,j;
	double r;
	const double  G =  0.00667384;
	for(i=0; i<gran; i++){
		r=0; 
		for(j=0; j < gran; j++){
			r = sqrt((rcvBodies[j].x-bodies[i].x)*(rcvBodies[j].x-bodies[i].x)+
				     (rcvBodies[j].y-bodies[i].y)*(rcvBodies[j].y-bodies[i].y));
			f[i].x += G*bodies[i].m*rcvBodies[j].m*(rcvBodies[j].x-bodies[i].x)/(r*r*r);
			f[i].y += G*bodies[i].m*rcvBodies[j].m*(rcvBodies[j].y-bodies[i].y)/(r*r*r);
		}
	}
}

void posUpdt(const int gran, body *bodies, fvp *f, fvp *v){
	int i;
	const double dt= 1;
	for(i=0; i<gran; i++){
		v[i].x += f[i].x * dt / bodies[i].m;
		v[i].y += f[i].y * dt / bodies[i].m;
		bodies[i].x += v[i].x * dt;
		bodies[i].y += v[i].y * dt;
	}
}

void scatter(const char *pName, int rank, int size, const int gran, body *bodies){
	int i, j;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;
	
	MPI_Datatype bodytype;
	MPI_Type_contiguous(3, MPI_DOUBLE, &bodytype);
	MPI_Type_commit(&bodytype);
	MPI_Status status;
	
	body *outbuf = (body *) malloc(gran*sizeof(body));
	if(rank==0){
		FILE *pFile;
		pFile = fopen(pName, "rb");
		for(j=0; j<gran; j++)
			fscanf(pFile,"%lf %lf %lf", &bodies[j].x, &bodies[j].y, &bodies[j].m);					
		for(i=0; i<size-rank-1; i++){
			for(j=0; j<gran; j++)
				fscanf(pFile,"%lf %lf %lf", &outbuf[j].x, &outbuf[j].y, &outbuf[j].m);
			MPI_Send(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD);
		}	
		fclose(pFile);
	}	
	else{
		MPI_Recv(bodies, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD, &status);
		for(i=0; i<size-rank-1; i++){
			MPI_Recv(outbuf, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD, &status);
			MPI_Send(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD);
		}	
	}
	free(outbuf);
}

void simulate(const int tmax, int rank, int size, const int gran, body *bodies){
	int t=tmax, i, round;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;
	
	MPI_Datatype bodytype;
	MPI_Type_contiguous(3, MPI_DOUBLE, &bodytype);
	MPI_Type_commit(&bodytype);
	MPI_Status status;

	MPE_XGraph bodyWin;	
	int winSize = 800;
	
	body *inbuf = (body *) malloc(gran*sizeof(body));
	body *outbuf = (body *) malloc(gran*sizeof(body));
	fvp *f = (fvp *) malloc(gran*sizeof(fvp));
	fvp *v = (fvp *) malloc(gran*sizeof(fvp));
	for(i=0; i < gran; i++){
		v[i].x=0;
		v[i].y=0;
	}

	MPE_Open_graphics(&bodyWin,MPI_COMM_WORLD,NULL,-1,-1,winSize,winSize,0);

	while(1){
		--t;
		round=size;
		memcpy(outbuf, bodies, gran*sizeof(body));		
		frcInit(gran, bodies, f);
		while (round > 1) {
			--round;
			if (!(rank % 2)){
				MPI_Send(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD);
				MPI_Recv(inbuf, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD, &status);						
			}
			else
			{
				MPI_Recv(inbuf, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD, &status);
				MPI_Send(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD);
			}
			memcpy(outbuf, inbuf, gran*sizeof(body));
			frcUpdt(gran, bodies, inbuf, f);
		}
		posUpdt(gran, bodies, f, v);
		for(i=0; i<gran; i++)
			MPE_Draw_circle(bodyWin,bodies[i].x,bodies[i].y,2,(rank*gran+i)%15+1);
		MPE_Update(bodyWin);
	  	usleep(300000);
		if(t > 0) 
			for(i=0;i<gran;i++) 
				MPE_Draw_circle(bodyWin,bodies[i].x,bodies[i].y,2,MPE_WHITE);
		if(t==0) break;
	}
	MPE_Close_graphics(&bodyWin);
	free(inbuf);
	free(outbuf);
	free(f);
	free(v);
}

void gather(int rank, int size, const int gran, body *bodies){
	int i, j;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;
	
	MPI_Datatype bodytype;
	MPI_Type_contiguous(3, MPI_DOUBLE, &bodytype);
	MPI_Type_commit(&bodytype);
	MPI_Status status;
	
	body *outbuf = (body *) malloc(gran*sizeof(body));
	if (rank != 0) {
		//memcpy(outbuf, bodies, gran*sizeof(body));
		MPI_Send(bodies, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD);
		for(i=0; i<size-rank-1; i++){
			MPI_Recv(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD, &status);
			MPI_Send(outbuf, gran, bodytype, recvfrom, 0, MPI_COMM_WORLD);
		}
	}
	else {
		FILE *oFile;
		oFile = fopen("peval_out.txt", "w");
		//memcpy(outbuf, bodies, gran*sizeof(body));
		for(j=0; j<gran; j++)
			fprintf(oFile, "%15.10f %15.10f %15.10f\n", bodies[j].x, bodies[j].y, bodies[j].m);
		for(i=0; i<size-rank-1; i++){
			MPI_Recv(outbuf, gran, bodytype, sendto, 0, MPI_COMM_WORLD, &status);
			for(j=0; j<gran; j++)
				fprintf(oFile, "%15.10f %15.10f %15.10f\n", outbuf[j].x, outbuf[j].y, outbuf[j].m);
		}	
		fclose(oFile);
	}
	free(outbuf);
}
