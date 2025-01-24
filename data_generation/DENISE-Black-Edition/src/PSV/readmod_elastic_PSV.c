/*------------------------------------------------------------------------
 *  Read elastic model properties (vp,vs,density) from files  
 *
 *  D. Koehn
 *  Kiel, 24.07.2016
 *  ----------------------------------------------------------------------*/


#include "fd.h"

void readmod_elastic_PSV(float  **  rho, float **  pi, float **  u){

	extern int NX, NY, NXG, NYG,  POS[3], MYID, INVMAT1;
	extern int WRITEMOD;
	extern char  MFILE[STRING_SIZE];	
	extern FILE *FP;

		
	/* local variables */
	float rhov, muv, piv, vp, vs;
	int i, j, ii, jj;
	FILE *fp_vs, *fp_vp, *fp_rho;
	char filename[STRING_SIZE];




	   fprintf(FP,"\n...reading model information from modell-files...\n");
           
	   /* read density and seismic velocities */
	   /* ----------------------------------- */
	   if(INVMAT1==1){ 
	   fprintf(FP,"\t Vp:\n\t %s.vp\n\n",MFILE);
	   sprintf(filename,"%s.vp",MFILE);
	   fp_vp=fopen(filename,"r");
	   if (fp_vp==NULL) err(" Could not open model file for Vp ! ");


	   fprintf(FP,"\t Vs:\n\t %s.vs\n\n",MFILE);
	   sprintf(filename,"%s.vs",MFILE);
	   fp_vs=fopen(filename,"r");
	   if (fp_vs==NULL) err(" Could not open model file for Vs ! ");

	   fprintf(FP,"\t Density:\n\t %s.rho\n\n",MFILE);
	   sprintf(filename,"%s.rho",MFILE);
	   fp_rho=fopen(filename,"r");
	   if (fp_rho==NULL) err(" Could not open model file for densities ! ");
           }
	   
	   /* read density and Lame parameters */
	   /* ----------------------------------- */
	   if(INVMAT1==3){ 
	   fprintf(FP,"\t Lame parameter lambda:\n\t %s.lam\n\n",MFILE);
	   sprintf(filename,"%s.lam",MFILE);
	   fp_vp=fopen(filename,"r");
	   if (fp_vp==NULL) err(" Could not open model file for Lame parameter lambda ! ");


	   fprintf(FP,"\t Lame parameter mu:\n\t %s.vs\n\n",MFILE);
	   sprintf(filename,"%s.mu",MFILE);
	   fp_vs=fopen(filename,"r");
	   if (fp_vs==NULL) err(" Could not open model file for Lame parameter mu ! ");

	   fprintf(FP,"\t Density:\n\t %s.rho\n\n",MFILE);
	   sprintf(filename,"%s.rho",MFILE);
	   fp_rho=fopen(filename,"r");
	   if (fp_rho==NULL) err(" Could not open model file for densities ! ");
           }
	   

	/* loop over global grid */
		for (i=1;i<=NXG;i++){
			for (j=1;j<=NYG;j++){
			fread(&vp, sizeof(float), 1, fp_vp);
			fread(&vs, sizeof(float), 1, fp_vs);
			fread(&rhov, sizeof(float), 1, fp_rho);
				
			/* only the PE which belongs to the current global gridpoint 
			is saving model parameters in his local arrays */
				if ((POS[1]==((i-1)/NX)) && 
				    (POS[2]==((j-1)/NY))){
					ii=i-POS[1]*NX;
					jj=j-POS[2]*NY;
                                
				u[jj][ii]=vs;
                                rho[jj][ii]=rhov;
                                pi[jj][ii]=vp;
				
				}
			}
		}
	




	fclose(fp_vp);
	fclose(fp_vs);
	fclose(fp_rho);
	
	
	/* each PE writes his model to disk */
        if(WRITEMOD){

	   sprintf(filename,"%s.denise.pi",MFILE);
	   writemod(filename,pi,3);
 	   MPI_Barrier(MPI_COMM_WORLD);

	   if (MYID==0) mergemod(filename,3);
	
	   sprintf(filename,"%s.denise.mu",MFILE);
           writemod(filename,u,3);
	   MPI_Barrier(MPI_COMM_WORLD);
	                           
	   if (MYID==0) mergemod(filename,3);
	
	   sprintf(filename,"%s.denise.rho",MFILE);
	   writemod(filename,rho,3);
	   MPI_Barrier(MPI_COMM_WORLD);
	                        
	   if (MYID==0) mergemod(filename,3);

	}

}




