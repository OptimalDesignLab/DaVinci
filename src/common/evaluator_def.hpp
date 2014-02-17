/**
 * \file evaluator_def.hpp
 * \brief definition of Evaluator methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

namespace davinci {
//==============================================================================
template <typename NodeT, typename ScalarT>
void Evaluator<NodeT,ScalarT>::SetDimensions(
    const int& num_elems, const int& num_nodes_per_elem,
    const int& num_cub_points, const int& num_ref_basis, const int& spatial_dim,
    const int& num_pdes) {
  BOOST_ASSERT_MSG(num_elems > 0, "num_elems must be > 0");
  BOOST_ASSERT_MSG(num_nodes_per_elem > 0, "num_nodes_per_elem must be > 0");
  BOOST_ASSERT_MSG(num_cub_points > 0, "num_cub_points must be > 0");
  BOOST_ASSERT_MSG(num_ref_basis > 0, "num_ref_basis must be > 0");
  BOOST_ASSERT_MSG(spatial_dim > 0, "spatial_dim must be > 0");
  BOOST_ASSERT_MSG(num_pdes > 0, "num_pdes must be > 0");
  num_elems_ = num_elems;
  num_nodes_per_elem_ = num_nodes_per_elem;
  num_cub_points_ = num_cub_points;
  num_ref_basis_ = num_ref_basis;
  dim_ = spatial_dim;
  num_pdes_ = num_pdes;
}
//==============================================================================
template <typename T, int N>
RCP<FieldContainer<T> > GenerateView(
    ArrayRCP<T>& data, const int& offset, const Tuple<int,N>& dimensions) {
  int size = 1;
  for (int i = 0; i < N; i++) size *= dimensions[i];
  return Teuchos::rcp(new FieldContainer<T>(
      dimensions, data.persistingView(offset, size)));
}
//==============================================================================
template <typename T, int N>
RCP<FieldContainer<const T> > GenerateConstView(
    ArrayRCP<T>& data, const int& offset, const Tuple<int,N>& dimensions) {
  int size = 1;
  for (int i = 0; i < N; i++) size *= dimensions[i];
  return Teuchos::rcp(new FieldContainer<const T>(
      dimensions, data.persistingView(offset, size)));
}
//==============================================================================
} // namespace davinci
