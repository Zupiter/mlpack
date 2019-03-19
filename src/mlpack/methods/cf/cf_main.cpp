/**
 * @file cf_main.hpp
 * @author Mudit Raj Gupta
 *
 * Main executable to run CF.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>

#include "cf.hpp"
#include "cf_model.hpp"

#include <mlpack/methods/cf/decomposition_policies/batch_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/regularized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_complete_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_incomplete_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/bias_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svdplusplus_method.hpp>

#include <mlpack/methods/cf/normalization/combined_normalization.hpp>
#include <mlpack/methods/cf/normalization/item_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/no_normalization.hpp>
#include <mlpack/methods/cf/normalization/overall_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/user_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/z_score_normalization.hpp>

#include <mlpack/methods/cf/interpolation_policies/average_interpolation.hpp>
#include <mlpack/methods/cf/interpolation_policies/regression_interpolation.hpp>
#include <mlpack/methods/cf/interpolation_policies/similarity_interpolation.hpp>

#include <mlpack/methods/cf/neighbor_search_policies/cosine_search.hpp>
#include <mlpack/methods/cf/neighbor_search_policies/lmetric_search.hpp>
#include <mlpack/methods/cf/neighbor_search_policies/pearson_search.hpp>


using namespace mlpack;
using namespace mlpack::cf;
using namespace mlpack::amf;
using namespace mlpack::svd;
using namespace mlpack::util;
using namespace std;

// Document program.
PROGRAM_INFO("Collaborative Filtering",
    // Short description.
    "An implementation of several collaborative filtering (CF) techniques for "
    "recommender systems.  This can be used to train a new CF model, or use an"
    " existing CF model to compute recommendations.",
    // Long description.
    "This program performs collaborative "
    "filtering (CF) on the given dataset. Given a list of user, item and "
    "preferences (the " + PRINT_PARAM_STRING("training") + " parameter), "
    "the program will perform a matrix decomposition and then can perform a "
    "series of actions related to collaborative filtering.  Alternately, the "
    "program can load an existing saved CF model with the " +
    PRINT_PARAM_STRING("input_model") + " parameter and then use that model "
    "to provide recommendations or predict values."
    "\n\n"
    "The input matrix should be a 3-dimensional matrix of ratings, where the "
    "first dimension is the user, the second dimension is the item, and the "
    "third dimension is that user's rating of that item.  Both the users and "
    "items should be numeric indices, not names. The indices are assumed to "
    "start from 0."
    "\n\n"
    "A set of query users for which recommendations can be generated may be "
    "specified with the " + PRINT_PARAM_STRING("query") + " parameter; "
    "alternately, recommendations may be generated for every user in the "
    "dataset by specifying the " +
    PRINT_PARAM_STRING("all_user_recommendations") + " parameter.  In "
    "addition, the number of recommendations per user to generate can be "
    "specified with the " + PRINT_PARAM_STRING("recommendations") + " "
    "parameter, and the number of similar users (the size of the neighborhood) "
    "to be considered when generating recommendations can be specified with "
    "the " + PRINT_PARAM_STRING("neighborhood") + " parameter."
    "\n\n"
    "For performing the matrix decomposition, the following optimization "
    "algorithms can be specified via the " + PRINT_PARAM_STRING("algorithm") +
    " parameter: "
    "\n"
    " - 'RegSVD' -- Regularized SVD using a SGD optimizer\n"
    " - 'NMF' -- Non-negative matrix factorization with alternating least "
    "squares update rules\n"
    " - 'BatchSVD' -- SVD batch learning\n"
    " - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning\n"
    " - 'SVDCompleteIncremental' -- SVD complete incremental learning\n"
    " - 'BiasSVD' -- Bias SVD using a SGD optimizer\n"
    " - 'SVDPP' -- SVD++ using a SGD optimizer\n"
    "\n\n"
    "The following neighbor search algorithms can be specified via" +
    " the " + PRINT_PARAM_STRING("neighbor_search") + " parameter:"
    "\n"
    " - 'cosine'  -- Cosine Search Algorithm\n"
    " - 'euclidean'  -- Euclidean Search Algorithm\n"
    " - 'pearson'  -- Pearson Search Algorithm\n"
    "\n\n"
    "The following weight interpolation algorithms can be specified via" +
    " the " + PRINT_PARAM_STRING("interpolation") + " parameter:"
    "\n"
    " - 'average'  -- Average Interpolation Algorithm\n"
    " - 'regression'  -- Regression Interpolation Algorithm\n"
    " - 'similarity'  -- Similarity Interpolation Algorithm\n"
    "\n"
    "A trained model may be saved to with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "To train a CF model on a dataset " + PRINT_DATASET("training_set") + " "
    "using NMF for decomposition and saving the trained model to " +
    PRINT_MODEL("model") + ", one could call: "
    "\n\n" +
    PRINT_CALL("cf", "training", "training_set", "algorithm", "NMF",
        "output_model", "model") +
    "\n\n"
    "Then, to use this model to generate recommendations for the list of users "
    "in the query set " + PRINT_DATASET("users") + ", storing 5 "
    "recommendations in " + PRINT_DATASET("recommendations") + ", one could "
    "call "
    "\n\n" +
    PRINT_CALL("cf", "input_model", "model", "query", "users",
        "recommendations", 5, "output", "recommendations"),
    SEE_ALSO("Collaborative filtering tutorial", "@doxygen/cftutorial.html"),
    SEE_ALSO("Alternating Matrix Factorization tutorial",
        "@doxygen/amftutorial.html"),
    SEE_ALSO("Collaborative Filtering on Wikipedia",
        "https://en.wikipedia.org/wiki/Collaborative_filtering"),
    SEE_ALSO("Matrix factorization on Wikipedia",
        "https://en.wikipedia.org/wiki/Matrix_factorization_"
        "(recommender_systems)"),
    SEE_ALSO("Matrix factorization techniques for recommender systems (pdf)",
        "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.3234"
        "&rep=rep1&type=pdf"),
    SEE_ALSO("mlpack::cf::CFType class documentation",
        "@doxygen/classmlpack_1_1cf_1_1CFType.html"));

// Parameters for training a model.
PARAM_MATRIX_IN("training", "Input dataset to perform CF on.", "t");
PARAM_STRING_IN("algorithm", "Algorithm used for matrix factorization.", "a",
    "NMF");
PARAM_INT_IN("neighborhood", "Size of the neighborhood of similar users to "
    "consider for each query user.", "n", 5);
PARAM_INT_IN("rank", "Rank of decomposed matrices (if 0, a heuristic is used to"
    " estimate the rank).", "R", 0);
PARAM_MATRIX_IN("test", "Test set to calculate RMSE on.", "T");

// Offer the user the option to set the maximum number of iterations, and
// terminate only based on the number of iterations.
PARAM_INT_IN("max_iterations", "Maximum number of iterations. If set to zero, "
    "there is no limit on the number of iterations.", "N", 1000);
PARAM_FLAG("iteration_only_termination", "Terminate only when the maximum "
    "number of iterations is reached.", "I");
PARAM_DOUBLE_IN("min_residue", "Residue required to terminate the factorization"
    " (lower values generally mean better fits).", "r", 1e-5);

// Load/save a model.
PARAM_MODEL_IN(CFModel, "input_model", "Trained CF model to load.", "m");
PARAM_MODEL_OUT(CFModel, "output_model", "Output for trained CF model.", "M");

// Query settings.
PARAM_UMATRIX_IN("query", "List of query users for which recommendations should"
    " be generated.", "q");
PARAM_FLAG("all_user_recommendations", "Generate recommendations for all "
    "users.", "A");
PARAM_UMATRIX_OUT("output", "Matrix that will store output recommendations.",
    "o");
PARAM_INT_IN("recommendations", "Number of recommendations to generate for each"
    " query user.", "c", 5);

PARAM_INT_IN("seed", "Set the random seed (0 uses std::time(NULL)).", "s", 0);

//  Interpolation and Neighbor Search Algorithms
PARAM_STRING_IN("interpolation", "Algorithm used for weight interpolation.",
    "i", "average");

PARAM_STRING_IN("neighbor_search", "Algorithm used for neighbor search.",
    "S", "euclidean");

template <typename NeighborSearchType,
          typename InterpolationType>
void ComputeRecommendations(CFModel* cf,
                            const size_t numRecs,
                            arma::Mat<size_t>& recommendations)
{
  // Reading users.
  if (CLI::HasParam("query"))
  {
    // User matrix.
    arma::Mat<size_t> users =
        std::move(CLI::GetParam<arma::Mat<size_t>>("query"));
    if (users.n_rows > 1)
      users = users.t();
    if (users.n_rows > 1)
      Log::Fatal << "List of query users must be one-dimensional!"
                 << std::endl;

    Log::Info << "Generating recommendations for "
              << users.n_elem << " users."
              << endl;

    cf->GetRecommendations<NeighborSearchType, InterpolationType>
        (numRecs, recommendations, users.row(0).t());
  }
  else
  {
    Log::Info << "Generating recommendations for all users." << endl;
    cf->GetRecommendations<NeighborSearchType, InterpolationType>
        (numRecs, recommendations);
  }
}

template <typename NeighborSearchType>
void ComputeRecommendations(CFModel* cf,
                            const size_t numRecs,
                            arma::Mat<size_t>& recommendations)
{
  //  Verifying the Interpolation algorithms
  RequireParamInSet<string>("interpolation", { "average",
      "regression", "similarity" }, true, "unknown interpolation algorithm");

  //  Taking Interpolation Alternatives
  const string interpolationAlgorithm = CLI::GetParam<string>("interpolation");

  //  Determining the Interpolation Algorithm
  if (interpolationAlgorithm == "average")
  {
    ComputeRecommendations<NeighborSearchType, AverageInterpolation>
        (cf, numRecs, recommendations);
  }
  else if (interpolationAlgorithm == "regression")
  {
    ComputeRecommendations<NeighborSearchType, RegressionInterpolation>
        (cf, numRecs, recommendations);
  }
  else if (interpolationAlgorithm == "similarity")
  {
    ComputeRecommendations<NeighborSearchType, SimilarityInterpolation>
        (cf, numRecs, recommendations);
  }
}

void ComputeRecommendations(CFModel* cf,
                            const size_t numRecs,
                            arma::Mat<size_t>& recommendations)
{
  //  Verifying the Neighbor Search algorithms
  RequireParamInSet<string>("neighbor_search", { "cosine",
      "euclidean", "pearson" }, true, "unknown neighbor search algorithm");

  //  Taking Neighbor Search alternatives
  const string neighborSearchAlgorithm = CLI::GetParam<string>
      ("neighbor_search");


  // Determining the Neighbor Search Algorithms
  if (neighborSearchAlgorithm == "cosine")
  {
    ComputeRecommendations<CosineSearch>(cf, numRecs, recommendations);
  }
  else if (neighborSearchAlgorithm == "euclidean")
  {
    ComputeRecommendations<EuclideanSearch>(cf, numRecs, recommendations);
  }
  else if (neighborSearchAlgorithm == "pearson")
  {
    ComputeRecommendations<PearsonSearch>(cf, numRecs, recommendations);
  }
}

template <typename NeighborSearchType,
          typename InterpolationType>
void ComputeRMSE(CFModel* cf)
{
  // Now, compute each test point.
  arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

  // Assemble the combination matrix to get RMSE value.
  arma::Mat<size_t> combinations(2, testData.n_cols);
  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    combinations(0, i) = size_t(testData(0, i));
    combinations(1, i) = size_t(testData(1, i));
  }

  // Now compute the RMSE.
  arma::vec predictions;
  cf->Predict<NeighborSearchType, InterpolationType>
      (combinations, predictions);

  // Compute the root of the sum of the squared errors, divide by the number of
  // points to get the RMSE.  It turns out this is just the L2-norm divided by
  // the square root of the number of points, if we interpret the predictions
  // and the true values as vectors.
  const double rmse = arma::norm(predictions - testData.row(2).t(), 2) /
      std::sqrt((double) testData.n_cols);

  Log::Info << "RMSE is " << rmse << "." << endl;
}

template <typename NeighborSearchType>
void ComputeRMSE(CFModel* cf)
{
  //  Verifying the Interpolation algorithms
  RequireParamInSet<string>("interpolation", { "average",
      "regression", "similarity" }, true, "unknown interpolation algorithm");

  //  Taking Interpolation Alternatives
  const string interpolationAlgorithm = CLI::GetParam<string>("interpolation");

  if (interpolationAlgorithm == "average")
  {
    ComputeRMSE<NeighborSearchType, AverageInterpolation>(cf);
  }
  else if (interpolationAlgorithm == "regression")
  {
    ComputeRMSE<NeighborSearchType, RegressionInterpolation>(cf);
  }
  else if (interpolationAlgorithm == "similarity")
  {
    ComputeRMSE<NeighborSearchType, SimilarityInterpolation>(cf);
  }
}

void ComputeRMSE(CFModel* cf)
{
  //  Verifying the Neighbor Search algorithms
  RequireParamInSet<string>("neighbor_search", { "cosine",
      "euclidean", "pearson" }, true, "unknown neighbor search algorithm");

  //  Taking Neighbor Search alternatives
  const string neighborSearchAlgorithm = CLI::GetParam<string>
    ("neighbor_search");

  if (neighborSearchAlgorithm == "cosine")
  {
    ComputeRMSE<CosineSearch>(cf);
  }
  else if (neighborSearchAlgorithm == "euclidean")
  {
    ComputeRMSE<EuclideanSearch>(cf);
  }
  else if (neighborSearchAlgorithm == "pearson")
  {
    ComputeRMSE<PearsonSearch>(cf);
  }
}

void PerformAction(CFModel* c)
{
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
  {
    // Get parameters for generating recommendations.
    const size_t numRecs = (size_t) CLI::GetParam<int>("recommendations");

    // Get the recommendations.
    arma::Mat<size_t> recommendations;
    ComputeRecommendations(c, numRecs, recommendations);

    // Save the output.
    CLI::GetParam<arma::Mat<size_t>>("output") = recommendations;
  }

  if (CLI::HasParam("test"))
    ComputeRMSE(c);

  CLI::GetParam<CFModel*>("output_model") = c;
}

template<typename DecompositionPolicy, typename NormalizationType>
void PerformAction(arma::mat& dataset,
                   const size_t rank,
                   const size_t maxIterations,
                   const double minResidue)
{
  const size_t neighborhood = (size_t) CLI::GetParam<int>("neighborhood");
  CFModel* c = new CFModel();
  c->template Train<DecompositionPolicy, NormalizationType>(dataset, neighborhood, rank,
      maxIterations, minResidue, CLI::HasParam("iteration_only_termination"));

  PerformAction(c);
}

void AssembleFactorizerType(const std::string& algorithm,
                            const std::string& numalgo,
                            arma::mat& dataset,
                            const size_t rank)
{
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double minResidue = CLI::GetParam<double>("min_residue");

  if (algorithm == "NMF")
  {
    if(numalgo=="comb_norm")
      PerformAction<NMFPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<NMFPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<NMFPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<NMFPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<NMFPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<NMFPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "BatchSVD")
  {
    if(numalgo=="comb_norm")
      PerformAction<BatchSVDPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<BatchSVDPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<BatchSVDPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<BatchSVDPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<BatchSVDPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<BatchSVDPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);

  }
  else if (algorithm == "SVDIncompleteIncremental")
  {
    if(numalgo=="comb_norm")
      PerformAction<SVDIncompletePolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<SVDIncompletePolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<SVDIncompletePolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<SVDIncompletePolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<SVDIncompletePolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<SVDIncompletePolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "SVDCompleteIncremental")
  {
    if(numalgo=="comb_norm")
      PerformAction<SVDCompletePolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<SVDCompletePolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<SVDCompletePolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<SVDCompletePolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<SVDCompletePolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<SVDCompletePolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "RegSVD")
  {
    ReportIgnoredParam("min_residue", "Regularized SVD terminates only "
        "when max_iterations is reached");
    if(numalgo=="comb_norm")
      PerformAction<RegSVDPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<RegSVDPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<RegSVDPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<RegSVDPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<RegSVDPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<RegSVDPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "RandSVD")
  {
    ReportIgnoredParam("min_residue", "Randomized SVD terminates only "
        "when max_iterations is reached");
    if(numalgo=="comb_norm")
      PerformAction<RandomizedSVDPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<RandomizedSVDPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<RandomizedSVDPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<RandomizedSVDPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<RandomizedSVDPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<RandomizedSVDPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "BiasSVD")
  {
    ReportIgnoredParam("min_residue", "Bias SVD terminates only "
        "when max_iterations is reached");
    if(numalgo=="comb_norm")
      PerformAction<BiasSVDPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<BiasSVDPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<BiasSVDPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<BiasSVDPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<BiasSVDPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<BiasSVDPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
  else if (algorithm == "SVDPP")
  {
    ReportIgnoredParam("min_residue", "SVD++ terminates only "
        "when max_iterations is reached");
    if(numalgo=="comb_norm")
      PerformAction<SVDPlusPlusPolicy, combined_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="item_mean_norm")
      PerformAction<SVDPlusPlusPolicy, item_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="no_norm")
      PerformAction<SVDPlusPlusPolicy, no_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="overall_norm")
      PerformAction<SVDPlusPlusPolicy, overall_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="user_mean_norm")
      PerformAction<SVDPlusPlusPolicy, user_mean_normalization>(dataset, rank, maxIterations, minResidue);
    else if(numalgo=="z_score_norm")
      PerformAction<SVDPlusPlusPolicy, z_score_normalization>(dataset, rank, maxIterations, minResidue);
  }
}

static void mlpackMain()
{
  if (CLI::GetParam<int>("seed") == 0)
    math::RandomSeed(std::time(NULL));
  else
    math::RandomSeed(CLI::GetParam<int>("seed"));

  // Validate parameters.
  RequireOnlyOnePassed({ "training", "input_model" }, true);

  // Check that nothing stupid is happening.
  if (CLI::HasParam("query") || CLI::HasParam("all_user_recommendations"))
    RequireOnlyOnePassed({ "query", "all_user_recommendations" }, true);

  RequireAtLeastOnePassed({ "output", "output_model" }, false,
      "no output will be saved");
  if (!CLI::HasParam("query") && !CLI::HasParam("all_user_recommendations"))
    ReportIgnoredParam("output", "no recommendations requested");

  RequireParamInSet<string>("algorithm", { "NMF", "BatchSVD",
      "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD",
      "RandSVD", "BiasSVD", "SVDPP" }, true, "unknown algorithm");

  ReportIgnoredParam({{ "iteration_only_termination", true }}, "min_residue");

  RequireParamInSet<string>("normalgo", {"comb_norm", "item_mean_norm",
    "no_norm", "overall_norm", "user_mean_norm","z_score_norm"}, true,
    "unknown normalization algorithm");
  RequireParamValue<int>("recommendations", [](int x) { return x > 0; }, true,
        "recommendations must be positive");
  ReportIgnoredParam({{ "iteration_only_termination", true }}, "min_residue");
  // Either load from a model, or train a model.
  if (CLI::HasParam("training"))
  {
    // Train a model.
    // Validate Parameters.
    ReportIgnoredParam({{ "iteration_only_termination", true }}, "min_residue");
    RequireParamValue<int>("rank", [](int x) { return x >= 0; }, true,
        "rank must be non-negative");
    RequireParamValue<double>("min_residue", [](double x) { return x >= 0; },
        true, "min_residue must be non-negative");
    RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; }, true,
        "max_iterations must be non-negative");
    RequireParamValue<int>("neighborhood", [](int x) { return x > 0; }, true,
        "neighborhood must be positive");

    // Read from the input file.
    arma::mat dataset = std::move(CLI::GetParam<arma::mat>("training"));

    RequireParamValue<int>("neighborhood",
        [&dataset](int x) { return x <= max(dataset.row(0)) + 1; }, true,
        "neighborbood must be less than or equal to the number of users");

    // Recommendation matrix.
    arma::Mat<size_t> recommendations;

    // Get parameters.
    const size_t rank = (size_t) CLI::GetParam<int>("rank");

    // Perform decomposition to prepare for recommendations.
    Log::Info << "Performing CF matrix decomposition on dataset..." << endl;

    const string algo = CLI::GetParam<string>("algorithm");

    // Perform the factorization and do whatever the user wanted.
    AssembleFactorizerType(algo, numalgo, dataset, rank);
  }
  else
  {
    // Load from a model after validating parameters.
    RequireAtLeastOnePassed({ "query", "all_user_recommendations",
        "test" }, true);

    // Load an input model.
    CFModel* c = std::move(CLI::GetParam<CFModel*>("input_model"));

    PerformAction(c);
  }
}
