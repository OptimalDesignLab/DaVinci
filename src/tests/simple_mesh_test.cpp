/**
 * \file simple_mesh_test.cpp
 * \brief unit test for the SimpleMesh class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Time.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_BlockMap.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "simple_mesh.hpp"

using Teuchos::GlobalMPISession;
using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using davinci::SimpleMesh;

BOOST_AUTO_TEST_SUITE(SimpleMesh_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();  
  SimpleMesh Mesh(std::cout, comm);
}

BOOST_AUTO_TEST_CASE(BuildRectangularMesh) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  Nx = 20;
  Lx = 0.5;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
}

BOOST_AUTO_TEST_CASE(BuildTpetraMap) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  RCP<const Tpetra::BlockMap<SimpleMesh::LocIdxT,SimpleMesh::GlbIdxT> > map;
  const int num_pdes = 5;
  Mesh.BuildTpetraMap(num_pdes, map);
}

BOOST_AUTO_TEST_CASE(BuildMatrixGraph) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 2, Ny = 2;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  RCP<const Tpetra::BlockMap<SimpleMesh::LocIdxT,SimpleMesh::GlbIdxT> > map;
  const int num_pdes = 5;
  Mesh.BuildTpetraMap(num_pdes, map);
  
  RCP<Tpetra::BlockCrsGraph<int,int> > jac_graph;
  Mesh.BuildMatrixGraph(map, jac_graph);
  BOOST_CHECK_EQUAL(jac_graph->getNodeNumBlockEntries(), 41);
}

BOOST_AUTO_TEST_CASE(BuildLinearSystemWorkSets) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  typedef Intrepid::Basis<double, Intrepid::FieldContainer<double> > BasisT;
  RCP<BasisT> workset;
  const int num_pdes = 1;
  Mesh.BuildLinearSystemWorkSets(num_pdes, workset);
  //workset->DefineCubature(2);
  //workset->EvaluateBasis();
}

BOOST_AUTO_TEST_CASE(ElemToNode) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 0), 1);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 1), 2);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 2), 12);
}

BOOST_AUTO_TEST_CASE(CopyElemNodeCoords) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 2, Ny = 2;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  int batch_size = 2;
  std::div_t div_result = std::div(Mesh.get_num_elems()-1, batch_size);
  int num_batch = div_result.quot+1;
  std::cout << "batch_size = " << batch_size << "\n";
  std::cout << "num_batch = " << num_batch << "\n";
  std::cout << "remainder = " << div_result.rem+1 << "\n";
  int num_nodes_per_elem = 3;
  int dim = 2;
  Teuchos::ArrayRCP<double> node_coords(batch_size*num_nodes_per_elem*2);
  for (int bi = 0; bi < num_batch; bi++) {
    Mesh.CopyElemNodeCoords(node_coords, bi, batch_size, num_batch);
    int set_num_elems = batch_size;
    if (bi == num_batch-1)
      set_num_elems = num_batch - (bi*batch_size);
    for (int ielem = 0; ielem < set_num_elems; ielem++) {
      int k = bi*batch_size + ielem;
      for (int i = 0; i < num_nodes_per_elem; i++)
        for (int j = 0; j < dim; j++)
          BOOST_CHECK_CLOSE(node_coords[(ielem*num_nodes_per_elem+i)*dim + j],
                            Mesh.ElemNodeCoord(k, i, j), 1e-13);
    }
  }
}

BOOST_AUTO_TEST_CASE(CopyElemDOFIndices) {
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  SimpleMesh Mesh(std::cout, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 2, Ny = 2;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  int batch_size = 2;
  std::div_t div_result = std::div(Mesh.get_num_elems()-1, batch_size);
  int num_batch = div_result.quot+1;
  int num_nodes_per_elem = 3;
  int num_pdes = 3;
  Teuchos::ArrayRCP<SimpleMesh::LocIdxT>
      dof_indices(batch_size*num_nodes_per_elem*num_pdes);
  for (int bi = 0; bi < num_batch; bi++) {
    Mesh.CopyElemDOFIndices(dof_indices, bi, batch_size, num_batch);
    int set_num_elems = batch_size;
    if (bi == num_batch-1)
      set_num_elems = num_batch - (bi*batch_size);
    for (int ielem = 0; ielem < set_num_elems; ielem++) {
      int k = bi*batch_size + ielem;
      for (int i = 0; i < num_nodes_per_elem; i++)
        BOOST_CHECK_EQUAL(dof_indices[ielem*num_nodes_per_elem+i],
                          Mesh.ElemToNode(k,i));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
