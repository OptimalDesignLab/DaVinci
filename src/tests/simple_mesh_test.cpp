/**
 * \file simple_mesh_test.cpp
 * \brief unit test for the SimpleMesh class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include <Teuchos_Time.hpp>
#include "Intrepid_FieldContainer.hpp"
#include "simple_mesh.hpp"

using davinci::SimpleMesh;

BOOST_AUTO_TEST_SUITE(SimpleMesh_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  SimpleMesh Mesh(std::cout);
}

BOOST_AUTO_TEST_CASE(BuildRectangularMesh) {
  SimpleMesh Mesh(std::cout);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  Nx = 20;
  Lx = 0.5;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
}

BOOST_AUTO_TEST_CASE(ElemToNode) {
  SimpleMesh Mesh(std::cout);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 0), 1);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 1), 2);
  BOOST_CHECK_EQUAL( Mesh.ElemToNode(2, 2), 12);
}

BOOST_AUTO_TEST_CASE(CopyElemNodeCoords) {
  SimpleMesh Mesh(std::cout);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 1000, Ny = 1000;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);
  int batch_size = 9;
  std::div_t div_result = std::div(Mesh.get_num_elems()-1, batch_size);
  int num_batch = div_result.quot+1;
  std::cout << "batch_size = " << batch_size << "\n";
  std::cout << "num_batch = " << num_batch << "\n";
  std::cout << "remainder = " << div_result.rem+1 << "\n";
  int num_nodes_per_elem = 3;
  Intrepid::FieldContainer<double> coords(batch_size, num_nodes_per_elem, 2);
  for (int bi = 0; bi < num_batch; bi++)
    Mesh.CopyElemNodeCoords(coords, bi, batch_size, num_batch); 
}

BOOST_AUTO_TEST_SUITE_END()
