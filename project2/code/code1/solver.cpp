#include "solver.h"
#include "omp.h"

/***************************************************************************************************
solverControl
****************************************************************************************************
Flow control to prepare and solve the system.
***************************************************************************************************/
void femSolver::solverControl(inputSettings* argSettings, triMesh* argMesh)
{
    cout << endl << "=================== SOLUTION =====================" << endl;

    mesh = argMesh;
    settings = argSettings;
    int ne = mesh->getNe();


    // Calculating time taken to execute each major function to get an idea about indentity of maximum time consuming loops


double start1,end1;                                       // 1
start1=omp_get_wtime();

    for(int iec=0; iec<ne; iec++)
    {
        calculateJacobian(iec);
        calculateElementMatrices(iec);
    }

end1=omp_get_wtime();
double etime1=end1-start1;
cout<<endl<<"Time consumed by fucntions calculateJacobian() and calculateElementMatrices() together: "<<etime1<<" seconds"<<endl;


double start2,end2;                                       // 2
start2=omp_get_wtime();

    applyDrichletBC();

end2=omp_get_wtime();
double etime2=end2-start2;
cout<<endl<<"Time consumed by function applyDrichletBC(): "<<etime2<<" seconds"<<endl;


double start3,end3;                                       // 3
start3=omp_get_wtime();

    accumulateMass();

end3=omp_get_wtime();
double etime3=end3-start3;
cout<<endl<<"Time consumed by function accumulateMass(): "<<etime3<<" seconds"<<endl<<endl;


double start4,end4;                                       // 4
start4=omp_get_wtime();

    explicitSolver();

end4=omp_get_wtime();
double etime4=end4-start4;
cout<<endl<<"Time consumed by function explicitsolver(): "<<etime4<<" seconds"<<endl;


    return;
}

/***************************************************************************************************
calculateJacobian
****************************************************************************************************
Compute and store the jacobian for each element.
***************************************************************************************************/
void femSolver::calculateJacobian(const int e)
{
    int myNode;     // node number for the current node
    double x[nen];  // x values for all the nodes of an element
    double y[nen];  // y values for all the nodes of an element

    triMasterElement* ME = mesh->getME(0);  // for easy access to the master element
    double * xyz = mesh->getXyz();

    double J[2][2];     // Jacobian for the current element
    double detJ;        // Jacobian determinant for the current element
    double invJ[2][2];  // inverse of Jacobian for the current element
    double dSdX[3];     // dSdx on a GQ point
    double dSdY[3];     // dSdy on a GQ point

    // collect element node coordinates in x[3] and y[3] matrices

    for (int i=0; i<nen; i++)
    {
        myNode =  mesh->getElem(e)->getConn(i);
        x[i] = xyz[myNode*nsd+xsd];
        y[i] = xyz[myNode*nsd+ysd];
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
        node = mesh->getElem(e)->getConn(i);
        mesh->getNode(node)->addMass(M[i][i]);
    }

    // At this point we have the necessary K, M, F matrices as a member of femSolver object.
    // They must be hard copied to the corresponding triElement variables.

    for(int i=0; i<nen; i++)
    {
        node = mesh->getElem(e)->getConn(i);
        x = xyz[node * nsd + xsd];
        y = xyz[node * nsd + ysd];
        radius = sqrt(pow(x,2) + pow(y,2));
        if(radius < (rFlux - 1e-10))
        {
            mesh->getElem(e)->setF(i, F[i]);
        }
        mesh->getElem(e)->setM(i, M[i][i]);
        for(int j=0; j<nen; j++)
        {
            mesh->getElem(e)->setK(i,j,K[i][j]);
        }
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
    int const nn = mesh->getNn();
    double * T = mesh->getT();
    double * xyz = mesh->getXyz();
    double x, y, radius;
    double rOuter = 0.1;
    double temp;
    this->nnSolved += 0;
    //if any of the boundary conditions set to Drichlet BC
    if (settings->getBC(1)->getType()==1)
    {
        for(int i=0; i<nn; i++)
        {
            x = xyz[i*nsd+xsd];
            y = xyz[i*nsd+ysd];
            radius = sqrt(pow(x,2) + pow(y,2));
            if (abs(radius-rOuter) <= 1E-10)
            {
                if(settings->getBC(1)->getType()==1)
                {
                    mesh->getNode(i)->setBCtype(1);
                    T[i] = settings->getBC(1)->getValue1();
                }
            }
            else
            {
                this->nnSolved += 1;
            }
        }

    }

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
    int const ne = mesh->getNe();
    int const nIter = settings->getNIter();
    double const dT = settings->getDt();
    double TL[3], MTnewL[3];
    double * massG = mesh->getMassG();
    double * MTnew = mesh->getMTnew();
    double * T = mesh->getT();
    double massTmp, MTnewTmp;
    double MT;
    double Tnew;
    double partialL2error, globalL2error, initialL2error;
    double* M;          // pointer to element mass matrix
    double* F;          // pointer to element forcing vector
    double* K;          // pointer to element stiffness matrix
    triElement* elem;   // temporary pointer to hold current element
    triNode* pNode;     // temporary pointer to hold partition nodes
    double maxT, minT, Tcur;
    minT = std::numeric_limits<double>::max();
    maxT = std::numeric_limits<double>::min();



    double etime1=0;
    double etime2=0;
    double start1,end1,start2,end2;

    for (int iter=0; iter<nIter; iter++)                                                
    {
        // clear RHS MTnew
        for(int i=0; i<nn; i++)
        {
            MTnew[i] = 0;
        }

        // Evaluate right hand side at element level





    // Since function explicitsolver() took maximum amount of time to execute
    // Checking time consumed by each loop in this function

                                      
start1=omp_get_wtime();               //looop 4.1

#pragma omp parallel for default(shared) firstprivate(elem,M,F,K,TL,MTnewL) reduction(+:MTnew[:nn])     // parallelizing using reduction

        for(int e=0; e<ne; e++)                                                                         // MOST TIME CONSUMING LOOP
        {
            elem = mesh->getElem(e);
            M = elem->getMptr();
            F = elem->getFptr();
            K = elem->getKptr();
            for(int i=0; i<nen; i++)
            {
                TL[i] = T[elem->getConn(i)];
            }

            MTnewL[0] = M[0]*TL[0] + dT*(F[0]-(K[0]*TL[0]+K[1]*TL[1]+K[2]*TL[2]));
            MTnewL[1] = M[1]*TL[1] + dT*(F[1]-(K[3]*TL[0]+K[4]*TL[1]+K[5]*TL[2]));
            MTnewL[2] = M[2]*TL[2] + dT*(F[2]-(K[6]*TL[0]+K[7]*TL[1]+K[8]*TL[2]));

            // RHS is accumulated at local nodes
            MTnew[elem->getConn(0)] += MTnewL[0];
            MTnew[elem->getConn(1)] += MTnewL[1];
            MTnew[elem->getConn(2)] += MTnewL[2];
        }


end1=omp_get_wtime();
etime1=etime1+end1-start1;



        // Evaluate the new temperature on each node on partition level
        partialL2error = 0.0;
        globalL2error = 0.0;



                                     
start2=omp_get_wtime();                 //loop 4.2

#pragma omp parallel for default(shared) firstprivate(pNode,massTmp,MT,Tnew,T) reduction(+:partialL2error)      // parallelizing using reduction

        for(int i=0; i<nn; i++)                                                                                // SECOND MOST TIME CONSUMING LOOP
        {
            pNode = mesh->getNode(i);
            if(pNode->getBCtype() != 1)
            {
                massTmp = massG[i];
                MT = MTnew[i];
                Tnew = MT/massTmp;
                partialL2error += pow(T[i]-Tnew,2);
                T[i] = Tnew;
            }
        }

end2=omp_get_wtime();
etime2=etime2+end2-start2;




        globalL2error = sqrt(partialL2error/this->nnSolved);

        if(iter==0)
        {
            initialL2error = globalL2error;
            cout << "The initial error is: " << initialL2error << endl;
            cout << "Iter" << '\t' << "Time" << '\t' << "L2 Error" << '\t' << endl;
        }

        globalL2error = globalL2error / initialL2error;

        if(iter%1000==0)
        {
            cout << iter << '\t';
            cout << fixed << setprecision(5) << iter*dT << '\t';
            cout << scientific << setprecision(5) << globalL2error << endl;
        }
        if(globalL2error <= 1.0E-7)
        {
            cout << iter << '\t';
            cout << fixed << setprecision(5) << iter*dT << '\t';
            cout << scientific << setprecision(5) << globalL2error << endl;
            break;
        }
    }

cout<<endl<<"Time consumed by loop 1 in function explicitsolver(): "<<etime1<<" seconds"<<endl;
cout<<endl<<"Time consumed by loop 2 in function explicitsolver(): "<<etime2<<" seconds"<<endl;

    return;
}


/***************************************************************************************************
* void femSolver::accumulateMass()
***************************************************************************************************
*
**************************************************************************************************/
void femSolver::accumulateMass()
{
    int nn = mesh->getNn();
    double * massG = mesh->getMassG();

    for(int i=0; i<nn; i++)
    {
        massG[i] = mesh->getNode(i)->getMass();
    }


    return;
}
