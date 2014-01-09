/**
 * \file simple_mesh_test.cpp
 * \brief unit test for the SimpleMesh class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include <Teuchos_Time.hpp>
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

BOOST_AUTO_TEST_SUITE_END()
