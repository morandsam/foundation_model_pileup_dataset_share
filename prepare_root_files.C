#include "TROOT.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"
#include "TSystem.h"
#include <algorithm>
#include <iostream>

using namespace ROOT;

// Function to prepare balanced ROOT files based on JetJVT cut variable
// Low JVT are put first, then high JVT
// If shuffle is needed, it is done using shuffle_root.py
void prepare_root_files() {
    ROOT::DisableImplicitMT();

    TChain tree("analysis");
    tree.Add("/eos/user/c/cmorenom/GlobalEventInterpretation/data/user.cmorenom.TEST.data18_13TeV.00364485.JetInputsJVT_v2.noSkimmed_EXT0_ANALYSIS.root/user.cmorenom.45995229._000002.ANALYSIS.root");

    // Filter low/high
    RDataFrame df(tree);
    auto df_low  = df.Filter("JetJVT <= 0.1 && JetJVT >= 0.0 && JetJVT >= 0");
    auto df_high = df.Filter("JetJVT >= 0.9 && JetJVT <= 1.0 && JetJVT >= 0");

    Long64_t n_low  = *df_low.Count();
    Long64_t n_high = *df_high.Count();
    Long64_t n_keep = std::min(n_low, n_high);

    const char* out_low  = "/eos/user/s/smorand/root_cut/data_low.root";
    const char* out_high = "/eos/user/s/smorand/root_cut/data_high.root";
    const char* out_final = "/eos/user/s/smorand/root_cut/data_bal_2.root";

    // Save balanced subsets
    df_low.Range(n_keep).Snapshot("analysis", out_low);
    df_high.Range(n_keep).Snapshot("analysis", out_high);

    // -------------------------
    // Optionally merge sequentially with hadd
    // -------------------------
    TString cmd = TString::Format("hadd -f %s %s %s", out_final, out_low, out_high);
    gSystem->Exec(cmd);

    std::cout << "Balanced dataset saved to: " << out_final << std::endl;
}
