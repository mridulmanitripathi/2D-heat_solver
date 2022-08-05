#include "solver.h"

/***************************************************************************************************
solverControl
****************************************************************************************************
Flow control to prepare and solve the system.
***************************************************************************************************/
void femSolver::solverControl(inputSettings* argSettings, triMesh* argMesh)
{
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    if (mype==0) cout << endl << "=================== SOLUTION =====================" << endl;

    mesh = argMesh;
    settings = argSettings;
    int nec = mesh->getNec();

    for(int iec=0; iec<nec; iec++)
    {
        calculateJacobian(iec);
        calculateElementMatrices(iec);
    }
    applyDrichletBC();
    accumulateMass();
    explicitSolver();

    return;
}

/***************************************************************************************************
calculateJacobian
****************************************************************************************************
Compute and store the jacobian for each element.
***************************************************************************************************/
void femSolver::calculateJacobian(const int e)
{
    int myNode;       // node number for the current node
    double x[nen];    // x values for all the nodes of an element
    double y[nen];    // y values for all the nodes of an element

    triMasterElement* ME = mesh->getME(0);    // for easy access to the master element

    double J[2][2];              // Jacobian for the current element
    double detJ;                 // Jacobian determinant for the current element
    double invJ[2][2];           // inverse of Jacobian for the current element
    double dSdX[3];              // dSdx on a GQ point
    double dSdY[3];              // dSdy on a GQ point

    // collect element node coordinates in x[3] and y[3] matrices
    for (int i=0; i<nen; i++)
    {
        myNode =  mesh->getElem(e)->getLConn(i);
        x[i] = mesh->getLNode(myNode)->getX();
        y[i] = mesh->getLNode(myNode)->getY();
    }

    // for all GQ points detJ, dSDx[3] and dSdY[3] are determined.
    for (int p=0; p<nGQP; p++)
    {
        // Calculate Jacobian
        J[0][0] = ME[p].getDSdKsi(0)*x[0] + ME[p].getDSdKsi(1)*x[1] + ME[p].getDSdKsi(2)*x[2];
        J[0][1] = ME[p].getDSdKsi(0)*y[0] + ME[p].getDSdKsi(1)*y[1] + ME[p].getDSdKsi(2)*y[2];
        J[1][0] = ME[p].getDSdEta(0)*x[0] + ME[p].getDSdEta(1)*x[1] + ME[p].getDSdEta(2)*x[2];
        J[1][1] = ME[p].getDSdEta(0)*y[0] + ME[p].getDSdEta(1)*y[1] + ME[p].getDSdEta(2)*y[2];

        //Calculate determinant of Jacobian and store in mesh
        detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
        mesh->getElem(e)->setDetJ(p, detJ);

        // Calculate inverse of Jacobian
        invJ[0][0] =  J[1][1]/detJ;
        invJ[0][1] = -J[0][1]/detJ;
        invJ[1][0] = -J[1][0]/detJ;
        invJ[1][1] =  J[0][0]/detJ;

        // Calculate dSdx and dSdy and store in mesh
        dSdX[0] = invJ[0][0]*ME[p].getDSdKsi(0) + invJ[0][1]*ME[p].getDSdEta(0);
        dSdX[1] = invJ[0][0]*ME[p].getDSdKsi(1) + invJ[0][1]*ME[p].getDSdEta(1);
        dSdX[2] = invJ[0][0]*ME[p].getDSdKsi(2) + invJ[0][1]*ME[p].getDSdEta(2);
        dSdY[0] = invJ[1][0]*ME[p].getDSdKsi(0) + invJ[1][1]*ME[p].getDSdEta(0);
        dSdY[1] = invJ[1][0]*ME[p].getDSdKsi(1) + invJ[1][1]*ME[p].getDSdEta(1);
        dSdY[2] = invJ[1][0]*ME[p].getDSdKsi(2) + invJ[1][1]*ME[p].getDSdEta(2);

        mesh->getElem(e)->setDSdX(p, 0, dSdX[0]);
        mesh->getElem(e)->setDSdX(p, 1, dSdX[1]);
        mesh->getElem(e)->setDSdX(p, 2, dSdX[2]);
        mesh->getElem(e)->setDSdY(p, 0, dSdY[0]);
        mesh->getElem(e)->setDSdY(p, 1, dSdY[1]);
        mesh->getElem(e)->setDSdY(p, 2, dSdY[2]);
    }

    return;
}

/***************************************************************************************************
void femSolver::calculateElementMatrices(const int e)
****************************************************************************************************
Compute the K, M and F matrices. Then accumulate the total mass into the node structure.
***************************************************************************************************/
void femSolver::calculateElementMatrices(const int e)
{
    int node;
    int D = settings->getD();
    double f = settings->getSource();
    double * xyz = mesh->getXyz();

    double totalM = 0.0;    // Total mass
    double totalDM = 0.0;   // Total diagonal mass
    double K[3][3];
    double M[3][3];
    double F[3];
    double x, y, radius;
    double rFlux = 0.01;

    // First, fill M, K, F matrices with zero for the current element
    for(int i=0; i<nen; i++)
    {
        F[i] = 0.0;
        mesh->getElem(e)->setF(i, 0.0);
        mesh->getElem(e)->setM(i, 0.0);
        for(int j=0; j<nen; j++)
        {
            mesh->getElem(e)->setK(i, j, 0.0);
            K[i][j] = 0.0;
            M[i][j] = 0.0;
        }
    }

    // Now, calculate the M, K, F matrices
    for(int p=0; p<nGQP; p++)
    {
        for(int i=0; i<nen; i++)
        {
            for(int j=0; j<nen; j++)
            {
                // Consistent mass matrix
                M[i][j] = M[i][j] +
                            mesh->getME(p)->getS(i) * mesh->getME(p)->getS(j) *
                            mesh->getElem(e)->getDetJ(p) * mesh->getME(p)->getWeight();
                // Stiffness matrix
                K[i][j] = K[i][j] +
                            D * mesh->getElem(e)->getDetJ(p) * mesh->getME(p)->getWeight() *
                            (mesh->getElem(e)->getDSdX(p,i) * mesh->getElem(e)->getDSdX(p,j) +
                            mesh->getElem(e)->getDSdY(p,i) * mesh->getElem(e)->getDSdY(p,j));
            }
        // Forcing matrix
        F[i] = F[i] + f * mesh->getME(p)->getS(i) * mesh->getElem(e)->getDetJ(p) * mesh->getME(p)->getWeight();
        }
    }

    // For the explicit solution, it is necessary to have a diagonal mass matrix and for this,
    // lumping of the mass matrix is necessary. In order to lump the mass matrix, we first need to
    // calculate the total mass and the total diagonal mass:
    for(int i=0; i<nen; i++)
    {
        for(int j=0; j<nen; j++)
        {
            totalM = totalM + M[i][j];
            if (i==j)
                totalDM = totalDM + M[i][j];
        }
    }

    // Now the diagonal lumping can be done
    for(int i=0; i<nen; i++)
    {
        for(int j=0; j<nen; j++)
        {
            if (i==j)
                M[i][j] = M[i][j] * totalM / totalDM;
            else
                M[i][j] = 0.0;
        }
    }

    //Total mass at each node is accumulated on local node structure:
    for(int i=0; i<nen; i++)
    {
        node = mesh->getElem(e)->getLConn(i);
        mesh->getLNode(node)->addMass(M[i][i]);
    }

    // At this point we have the necessary K, M, F matrices as a member of femSolver object.
    // They must be hard copied to the corresponding triElement variables.
    for(int i=0; i<nen; i++)
    {
        node =  mesh->getElem(e)->getLConn(i);
        x = mesh->getLNode(node)->getX();
        y = mesh->getLNode(node)->getY();
        // node = mesh->getElem(e)->getLConn(i);
        // x = xyz[node * nsd + xsd];
        // y = xyz[node * nsd + ysd];
        radius = sqrt(pow(x,2) + pow(y,2));
        if(radius < (rFlux - 1e-10))
        {
            mesh->getElem(e)->setF(i, F[i]);
        }
        mesh->getElem(e)->setM(i, M[i][i]);
        for(int j=0; j<nen; j++)
        {
            mesh->getElem(e)->setK(i,j,K[i][j]);
//             cout << "K: " << K[i][j] << "\t";
        }
//        cout << "\n";
    }
    return;
}

/***************************************************************************************************
void femSolver::applyDrichletBC()
****************************************************************************************************
if any of the boundary conditions set to Drichlet type
    visits all partition level nodes
        determines if any the nodes is on any of the side surfaces

***************************************************************************************************/
void femSolver::applyDrichletBC()
{
    int const nnc = mesh->getNnc();
    double * TG = mesh->getTG();
    double * xyz = mesh->getXyz();
    double x, y, radius;
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    // double rInner = 0.2;
    double rOuter = 0.1;
    double temp;
    this->nnSolved = 0;
    int partialnnSolved = 0;
    //if any of the boundary conditions set to Drichlet BC
    if (settings->getBC(1)->getType()==1)
    {
        for(int i=0; i<nnc; i++)
        {
            x = xyz[i*nsd+xsd];
            y = xyz[i*nsd+ysd];
            radius = sqrt(pow(x,2) + pow(y,2));
            if (abs(radius-rOuter) <= 1E-10)
            {
                if(settings->getBC(1)->getType()==1)
                {
                    mesh->getNode(i)->setBCtype(1);
                    TG[i] = settings->getBC(1)->getValue1();
                }
            }
            else
            {
                partialnnSolved += 1;
            }
        }
    }
    MPI_Allreduce(&partialnnSolved, &this->nnSolved, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // if (mype == 0) cout << "nnSolved: " << this->nnSolved << endl;

    return;
}

/***************************************************************************************************
 * void femSolver::explicitSolver()
 ***************************************************************************************************
 *
 **************************************************************************************************/
void femSolver::explicitSolver()
{
    int const nn = mesh->getNn();
    int const nec = mesh->getNec();
    int const nnc = mesh->getNnc();
    int const nnl = mesh->getNnl();
    int const mnc = mesh->getMnc();
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    int const nIter = settings->getNIter();
    double const dT = settings->getDt();
    double T[3], MTnew[3];
    double mass;
    double * massG = mesh->getMassG();
    double * MTnewG = mesh->getMTnewG();
    double * MTnewL = mesh->getMTnewL();
    double * TG = mesh->getTG();
    double * TL = mesh->getTL();
    double massTmp, MTnewTmp;
    double MT;
    double Tnew;
    double partialL2error, globalL2error, initialL2error;
    double* M;                   // pointer to element mass matrix
    double* F;                   // pointer to element forcing vector
    double* K;                   // pointer to element stiffness matrix
    triElement* elem;            // temporary pointer to hold current element
    triNode* pNode;              // temporary pointer to hold partition nodes
    double maxT, minT, Tcur;
    minT = std::numeric_limits<double>::max();
    maxT = std::numeric_limits<double>::min();
    int nodeG[nen];
    double partialsumT = 0;
    double sumT;

    // Complete MPI Communication here

    MPI_Win winMTnew;            // Window for MTnew
    MPI_Win winTG;               // Window for TG
    MPI_Win_create(MTnewG,mnc*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&winMTnew);
    MPI_Win_create(TG,mnc*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&winTG);

    // Add MPI Communication here

    localizeTemperature(winTG);     // To get TL

    for (int iter=0; iter<nIter; iter++)
    {
        // clear RHS MTnew at local node level
        for(int i=0; i<nnl; i++){
            MTnewL[i] = 0;
        }

        // Evaluate right hand side at element level
        for(int e=0; e<nec; e++)
        {
            elem = mesh->getElem(e);
            M = elem->getMptr();
            F = elem->getFptr();
            K = elem->getKptr();
            for(int i=0; i<nen; i++)
            {
                nodeG[i] = elem->getLConn(i);
                T[i] = TL[nodeG[i]];
            }

            MTnew[0] = M[0]*T[0] + dT*(F[0]-(K[0]*T[0]+K[1]*T[1]+K[2]*T[2]));
            MTnew[1] = M[1]*T[1] + dT*(F[1]-(K[3]*T[0]+K[4]*T[1]+K[5]*T[2]));
            MTnew[2] = M[2]*T[2] + dT*(F[2]-(K[6]*T[0]+K[7]*T[1]+K[8]*T[2]));

            // RHS is accumulated at local nodes
            MTnewL[nodeG[0]] += MTnew[0];
            MTnewL[nodeG[1]] += MTnew[1];
            MTnewL[nodeG[2]] += MTnew[2];
        }

        // Add MPI Communication here

        accumulateMTnew(winMTnew);          // To get MTnewG

        // Evaluate the new temperature on each node on partition level
        partialL2error = 0.0;
        globalL2error = 0.0;
        for(int i=0; i<nnc; i++)
        {
            pNode = mesh->getNode(i);
            if(pNode->getBCtype() != 1)
            {
                massTmp = massG[i];
                // cout << "mass: " << massTmp << "\n";
                MT = MTnewG[i];
                // cout << "MTnew: " << MT << "\n";
                Tnew = MT/massTmp;
                // cout << "Tnew: " << Tnew << "\n";
                partialL2error += pow(TG[i]-Tnew,2);
                TG[i] = Tnew;
                MTnewG[i] = 0;
            }
        }

        // Add MPI Communication here

        localizeTemperature(winTG);         // To get TL

        // Reduce RMSerrors to all PEs
        MPI_Allreduce(&partialL2error, &globalL2error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        globalL2error = sqrt(globalL2error/(double)this->nnSolved);

        if(iter==0)
        {
            initialL2error = globalL2error;
            if (mype==0){
                cout << "The initial error is: " << scientific << setprecision(15) << initialL2error << endl;
                cout << "Iter" << '\t' << "Time" << '\t' << "L2 Error" << '\t' << endl;
            }
        }
        globalL2error = globalL2error / initialL2error;

        if(mype==0 && iter%1000==0)
        {
            cout << iter << '\t';
            cout << fixed << setprecision(5) << iter*dT << '\t';
            cout << scientific << setprecision(5) << globalL2error << endl;
        }
        if(globalL2error <= 1.0E-7)
        {
            if (mype==0)
            {
                cout << iter << '\t';
                cout << fixed << setprecision(5) << iter*dT << '\t';
                cout << scientific << setprecision(5) << globalL2error << endl;
            }
            break;
        }
    }

    // Add MPI Communication here

    MPI_Win_free(&winMTnew);        // To free the created window
    MPI_Win_free(&winTG);

    // Sum of the temperature array for debugging purpos
    partialsumT = 0.0;
    for (int i = 0; i < nnc; i++)
    {
        partialsumT += TG[i];
    }
    MPI_Allreduce(&partialsumT, &sumT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (mype == 0) cout << "sumT " << sumT << endl;

    return;
}

/***************************************************************************************************
 * void femSolver::accumulateMass()
 ***************************************************************************************************
 *
 **************************************************************************************************/
void femSolver::accumulateMass()
{
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    //ADD CODE HERE

    int cnnc,proc,in;

    int mnc=mesh->getMnc();
    int nnc=mesh->getNnc();
    int nn=mesh->getNn();
    int nnl=mesh->getNnl();
    double *massG=mesh->getMassG();

    int nodeLToG;                                       // Stores global node number

    double *temp;
    temp=new double[mnc*sizeof(double)];                // Dynamic memory allocation of temp array

    MPI_Win win;                              		// Creating window for global mass
    MPI_Win_create(massG,mnc*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&win);
    MPI_Win_fence(0,win);

    for(int i=0;i<npes;i++)
        {
        for(int j=0;j<mnc;j++)
            {temp[j]=0;}

        if(npes==1)                           		// Calculating nnc for each processor
            {cnnc=nn;}
        else
            {
            if((mnc*i)>(nn-mnc))
                {cnnc=max(0,nn-mnc*i);}
            else
                {cnnc=mnc;}
            }

        for(int l=0;l<nnl;l++)
            {
            nodeLToG=mesh->getNodeLToG()[l];     	 // Storing global node number of corresponding local node
            proc=nodeLToG/mnc;             		 // Determining processor rank of the local node

            if(proc==i)                    		 // For current pe
                {
                in=(nodeLToG)%mnc;         		 // Determining index
                temp[in]=mesh->getLNode(l)->getMass();
                }
            }

        MPI_Win_fence(0,win);               		 // Value in temp is added for each processor
        MPI_Accumulate(temp,cnnc,MPI_DOUBLE,i,0,cnnc,MPI_DOUBLE,MPI_SUM,win);
        MPI_Win_fence(0,win);
        }
    free(temp);

    return;
}

/***************************************************************************************************
 * void femSolver::accumulateMTnew()
 ***************************************************************************************************
 *
 **************************************************************************************************/
void femSolver::accumulateMTnew(MPI_Win win)
{
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    //ADD CODE HERE

    double *MTnewL=mesh->getMTnewL();

    int mnc=mesh->getMnc();
    int nnc=mesh->getNnc();
    int nn=mesh->getNn();
    int nnl=mesh->getNnl();

    int cnnc, proc,in;

    double *temp;
    temp=new double[mnc*sizeof(double)];             // Dynamic memory allocation of temp array

    int nodeLToG;                                    // Stores global node number

    for(int i=0;i<npes;i++)
        {
        for(int j=0;j<mnc;j++)
            {temp[j]=0;}

        if(npes==1)                                  // Determining nnc for each pe
            {cnnc=nn;}
        else
            {
            if((mnc*i)>(nn-mnc))
                {cnnc=max(0,nn-mnc*i);}
            else
                {cnnc=mnc;}
            }
        for(int l=0;l<nnl;l++)
            {
            nodeLToG=mesh->getNodeLToG()[l];        // Storing global node number of corresponding local node
            proc=(nodeLToG)/mnc;                    // Determining processor rank for the local node

            if(proc==i)
                {
                in=(nodeLToG)%mnc;                 // Determining index
                temp[in]=MTnewL[l];
                }
            }

        MPI_Win_fence(0,win);                      // Value in temp is added for each pe
        MPI_Accumulate(temp,cnnc,MPI_DOUBLE,i,0,cnnc,MPI_DOUBLE,MPI_SUM,win);
        MPI_Win_fence(0,win);
        }
    free(temp);

    return;
}

/***************************************************************************************************
 * void femSolver::localizeTemperature()
 ***************************************************************************************************
 *
 **************************************************************************************************/
void femSolver::localizeTemperature(MPI_Win win)
{
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    //ADD CODE HERE

    double *TL=mesh->getTL();

    int mnc=mesh->getMnc();
    int nn=mesh->getNn();
    int nnl=mesh->getNnl();

    int cnnc,proc,in;

    double *temp;
    temp=new double[mnc*sizeof(double)];            // Dynamic memory allocation of temp array

    int nodeLToG;                                   // Stores global node number

    for(int i=0;i<npes;i++)
        {
        for(int j=0;j<mnc;j++)
            {temp[j]=0;}

        if(npes==1)                                 // Determining nnc for each pe
            {cnnc=nn;}
        else
            {
            if((mnc*i)>(nn-mnc))
                {cnnc=max(0,nn-mnc*i);}
            else
                {cnnc=mnc;}
            }

        MPI_Win_fence(0,win);                       // temp stores the values of TG
        MPI_Get(temp,cnnc,MPI_DOUBLE,i,0,cnnc,MPI_DOUBLE,win);
        MPI_Win_fence(0,win);

        for(int l=0;l<nnl;l++)
            {
            nodeLToG=mesh->getNodeLToG()[l];        // Storing global node number of corresponding local node
            proc=(nodeLToG)/mnc;                    // Determining processor rank for the local node

            if(proc==i)                             // For current pe
                {
                in=(nodeLToG)%mnc;                  // Determining index
                TL[l]=temp[in];
                }
            }
        }
    free(temp);

    return;
}
