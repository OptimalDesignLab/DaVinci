/**
 * \file work_set_test.cpp
 * \brief unit test for the WorkSet<NodeT,ScalarT,MeshT> class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_Time.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "work_set.hpp"
#include "simple_mesh.hpp"
#include "metric_jacobian.hpp"
#include "laplace.hpp"

using Teuchos::GlobalMPISession;
using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using Intrepid::FieldContainer;
using davinci::WorkSet;
using davinci::SimpleMesh;
using davinci::Evaluator;
using davinci::MetricJacobian;
using davinci::Laplace;

typedef Tpetra::Vector<double,int,int> Vector;
typedef Tpetra::CrsMatrix<double,int,int> Matrix;
typedef Tpetra::Map<int,int> Map;
//typedef Tpetra::BlockMap<int,int> BlockMap;

BOOST_AUTO_TEST_SUITE(WorkSet_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
}

BOOST_AUTO_TEST_CASE(Topology) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  cell = Teuchos::rcp(
      shards::getCellTopologyData<shards::Tetrahedron<4> >(), false);
  MyWorkSet.DefineTopology(cell);
}

BOOST_AUTO_TEST_CASE(Cubature) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  for (int deg = 1; deg < 10; deg++)
    MyWorkSet.DefineCubature(deg);
}

BOOST_AUTO_TEST_CASE(Basis) {
  typedef double ScalarT;
  WorkSet<ScalarT,ScalarT,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<ScalarT, FieldContainer<ScalarT>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);
}

BOOST_AUTO_TEST_CASE(Evaluators) {
  typedef double ScalarT;
  typedef double NodeT;
  WorkSet<NodeT,ScalarT,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  std::list<Evaluator<NodeT,ScalarT>* > evaluators;
  evaluators.push_back(new MetricJacobian<NodeT,ScalarT>());
  evaluators.push_back(new Laplace<NodeT,ScalarT>());
  MyWorkSet.DefineEvaluators(evaluators);
}

BOOST_AUTO_TEST_CASE(ResizeSets) {  
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);  
  std::list<Evaluator<double,double>* > evaluators;
  evaluators.push_back(new MetricJacobian<double,double>());
  evaluators.push_back(new Laplace<double,double>());
  MyWorkSet.DefineEvaluators(evaluators);
  int num_pdes = 1;
  int total_elems = 100;
  for (int nelems = 1; nelems <= total_elems; nelems++)
    MyWorkSet.ResizeSets(num_pdes, total_elems, nelems);
}

BOOST_AUTO_TEST_CASE(BuildSystem) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  //DefaultComm<int>::getComm();

  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  
  // Define a rectangular mesh  
  SimpleMesh Mesh(out);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);

  // Create a Workset for the Laplace PDE
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(out);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);
  std::list<Evaluator<double,double>* > evaluators;
  evaluators.push_back(new MetricJacobian<double,double>());
  evaluators.push_back(new Laplace<double,double>());
  MyWorkSet.DefineEvaluators(evaluators);
  int num_pdes = 1;
  int total_elems = Mesh.get_num_elems();
  int nelems = 10;
  MyWorkSet.ResizeSets(num_pdes, total_elems, nelems);
  
  // create Tpetra map and linear algebra objects
  const int index_base = 0; // where indexing starts, i.e. c-style
  const size_t num_rows = Mesh.get_num_nodes();
  const Tpetra::global_size_t num_global_rows = num_rows*comm->getSize();
  const int num_local = num_global_rows; // serial for now
  RCP<const Map> map =
      Tpetra::createContigMap<int,int>(num_global_rows, num_local, comm);
#if 0
  RCP<const BlockMap> block_map = new
      Tpetra::BlockMap(num_global_rows, 1, 0, comm);
#endif
  RCP<Vector> sol = Tpetra::createVector<double>(map);
  sol->putScalar(1.0);
  RCP<Vector> rhs = Tpetra::createVector<double>(map);
  RCP<Matrix> jacobian = Tpetra::createCrsMatrix<double,int,int>(map);
  MyWorkSet.BuildSystem(Mesh, sol, rhs, jacobian);
}

#if 0
BOOST_AUTO_TEST_CASE(CopyMeshCoords) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);

  SimpleMesh Mesh(std::cout);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 1000, Ny = 1000;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);

  int batch_size = 9;
  MyWorkSet.ResizeSets(Mesh.get_num_elems(), batch_size);
  std::div_t div_result = std::div(Mesh.get_num_elems()-1, batch_size);
  int num_batch = div_result.quot+1;
  std::cout << "batch_size = " << batch_size << "\n";
  std::cout << "num_batch = " << num_batch << "\n";
  std::cout << "remainder = " << div_result.rem+1 << "\n";
  for (int bi = 0; bi < num_batch; bi++)
    MyWorkSet.CopyMeshCoords(Mesh, bi);
  
  // Teuchos::Time workset_time("Time to access inside WorkSet: ");
  // Teuchos::Time direct_time("Time to access directly      : ");
  // workset_time.start();
  // MyWorkSet.BuildSystem(Mesh);
  // workset_time.stop();
  // direct_time.start();
  // for (int ielem = 0; ielem < Mesh.get_num_elems(); ielem++)
  //   for (int i = 0; i < 3; i++)
  //     int tmp = Mesh.ElemToNode(ielem, i);
  // direct_time.stop();
  // std::cout << workset_time.name() << workset_time.totalElapsedTime() << "\n";
  // std::cout << direct_time.name() << direct_time.totalElapsedTime() << "\n";
}
#endif

BOOST_AUTO_TEST_SUITE_END()
