// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2024 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Chrono Contributors
// =============================================================================
//
// Helmet blast-impact simulation using an equivalent SPH pressure loading.
//
// Scenario
// --------
//   A 14 kg C4 charge is detonated at a standoff distance of 6.1 m.  The
//   explosion source and air fluid domain are NOT modeled.  Instead, the
//   equivalent Friedlander incident overpressure history (derived from
//   Kingery-Bulmash scaling relations) is applied directly as a surface
//   force on the masseter (jaw-muscle) region of a rigid combat-helmet body.
//
// Physics
// -------
//   TNT equivalent mass  : W = 14 kg × 1.19 (C4/TNT) = 16.66 kg TNT
//   Hopkinson-Cranz scale: Z = R / W^(1/3)  [m kg^(-1/3)]
//   Friedlander waveform : P(t) = Pso*(1 - t/td)*exp(-b*t/td)  for 0 ≤ t ≤ td
//
//   Blast parameters for Z ≈ 2.39 m kg^(-1/3) (Kingery-Bulmash, free-air):
//     Pso  ≈ 80 kPa   (peak incident overpressure)
//     td   ≈ 7.66 ms  (positive-phase duration, scaled by W^(1/3))
//     b    ≈ 1.5      (Friedlander decay coefficient)
//
// Helmet model
// ------------
//   Rigid body approximated as a solid ellipsoid (280 × 240 × 200 mm) with
//   an effective mass of 1.5 kg and corresponding inertia tensor.  The blast
//   force is applied at the approximate masseter surface location on the
//   lateral (+Y) face of the helmet.
//
// References
// ----------
//   Kingery & Bulmash (1984) "Airblast Parameters from TNT Spherical Air
//     Burst and Hemispherical Surface Burst", BRL-TR-02555
//   UFC 3-340-02 (2008) "Structures to Resist the Effects of Accidental
//     Explosions"
//   Friedlander (1946) idealized blast-wave overpressure profile
// =============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChForce.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/functions/ChFunctionLambda.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;

// =============================================================================
// Blast parameters
// =============================================================================

// C4 charge and standoff
static constexpr double C4_MASS_KG  = 14.0;   ///< C4 charge mass [kg]
static constexpr double C4_TNT_FACTOR = 1.19;  ///< C4-to-TNT mass equivalence factor
static constexpr double STANDOFF_M  = 6.1;     ///< standoff distance [m]

// Friedlander waveform coefficients (Kingery-Bulmash scaled, free-air burst)
static constexpr double PSO_PA    = 80.0e3;   ///< peak incident overpressure [Pa]
static constexpr double TD_SCALED = 3.0e-3;   ///< scaled positive-phase duration [s kg^(-1/3)]
static constexpr double B_DECAY   = 1.5;      ///< Friedlander decay coefficient [-]

// Masseter-region surface area exposed to blast
static constexpr double A_MASSETER_M2 = 80.0e-4;  ///< ~80 cm² in m²

// =============================================================================
// Helmet physical properties (PASGT-style composite combat helmet)
// =============================================================================

static constexpr double HELMET_MASS_KG = 1.5;   ///< total helmet mass [kg]

// Approximate principal moments of inertia about the centre of mass for a thin
// ellipsoidal shell with semi-axes a=0.14 m, b=0.12 m, c=0.10 m [kg·m²]:
//   Ixx = (m/5) * (b² + c²),  Iyy = (m/5) * (a² + c²),  Izz = (m/5) * (a² + b²)
static constexpr double HELMET_IXX = (HELMET_MASS_KG / 5.0) * (0.12 * 0.12 + 0.10 * 0.10);
static constexpr double HELMET_IYY = (HELMET_MASS_KG / 5.0) * (0.14 * 0.14 + 0.10 * 0.10);
static constexpr double HELMET_IZZ = (HELMET_MASS_KG / 5.0) * (0.14 * 0.14 + 0.12 * 0.12);

// =============================================================================
// Helper: Friedlander blast pressure and equivalent force as functions of time
// =============================================================================

/// Returns the Friedlander incident overpressure [Pa] at time t [s].
/// The blast wave arrives at t = 0.  For t outside [0, td] the overpressure
/// is zero (negative phase is conservatively neglected).
static double BlastPressure(double t, double td) {
    if (t <= 0.0 || t >= td)
        return 0.0;
    const double tau = t / td;
    return PSO_PA * (1.0 - tau) * std::exp(-B_DECAY * tau);
}

/// Returns the equivalent blast force [N] at time t [s]:
///   F(t) = P(t) × A_masseter
static double BlastForce(double t, double td) {
    return BlastPressure(t, td) * A_MASSETER_M2;
}

// =============================================================================
int main(int argc, char* argv[]) {
    // -------------------------------------------------------------------------
    // Compute blast parameters from Kingery-Bulmash scaling
    // -------------------------------------------------------------------------
    const double W_tnt   = C4_MASS_KG * C4_TNT_FACTOR;          // [kg TNT]
    const double W_cbrt  = std::cbrt(W_tnt);                     // W^(1/3)
    const double Z       = STANDOFF_M / W_cbrt;                  // scaled distance [m kg^(-1/3)]
    const double td      = TD_SCALED * W_cbrt;                   // positive-phase duration [s]
    const double F_peak  = BlastForce(1e-12, td);            // peak force (at t→0⁺) [N]

    std::cout << "=== Helmet Blast Impact Simulation (Chrono SPH Equivalent Pressure) ===" << std::endl;
    std::cout << "  C4 charge mass      : " << C4_MASS_KG  << " kg"   << std::endl;
    std::cout << "  TNT equivalent      : " << W_tnt        << " kg"   << std::endl;
    std::cout << "  Standoff distance   : " << STANDOFF_M   << " m"    << std::endl;
    std::cout << "  Scaled distance Z   : " << Z            << " m/kg^(1/3)" << std::endl;
    std::cout << "  Peak overpressure   : " << PSO_PA / 1e3 << " kPa"  << std::endl;
    std::cout << "  Positive phase td   : " << td * 1e3     << " ms"   << std::endl;
    std::cout << "  Masseter area       : " << A_MASSETER_M2 * 1e4 << " cm²" << std::endl;
    std::cout << "  Peak blast force    : " << F_peak       << " N"    << std::endl;
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // Simulation settings
    // -------------------------------------------------------------------------
    const double step_size = 1.0e-6;      // time step [s] — micro-second resolution
    const double t_end     = 3.0 * td;    // simulate 3 positive-phase durations

    // -------------------------------------------------------------------------
    // Output directory
    // -------------------------------------------------------------------------
    const std::string out_dir = GetChronoOutputPath() + "FSI_HelmetBlast";
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cerr << "Error creating output directory: " << out_dir << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // Create the Chrono multi-body system
    // -------------------------------------------------------------------------
    ChSystemNSC sys;
    sys.SetGravitationalAcceleration(ChVector3d(0, 0, -9.81));

    // -------------------------------------------------------------------------
    // Create the helmet rigid body
    //
    // Geometry: ellipsoid with full-axis dimensions 280 mm × 240 mm × 200 mm.
    // The density fed to ChBodyEasyEllipsoid is chosen so that the auto-
    // calculated mass matches the real helmet mass; inertia is overridden
    // afterwards with the thin-shell approximation.
    //
    // Axis convention (body frame):
    //   X — fore-aft (positive forward)
    //   Y — lateral  (positive to the right — blast-wave direction)
    //   Z — superior (positive upward)
    // -------------------------------------------------------------------------
    const ChVector3d helmet_axes(0.28, 0.24, 0.20);  // full axis lengths [m]

    // Volume of the ellipsoid = π/6 * ax * ay * az
    const double ellipsoid_volume = (CH_PI / 6.0) * helmet_axes.x() * helmet_axes.y() * helmet_axes.z();
    const double effective_density = HELMET_MASS_KG / ellipsoid_volume;  // [kg/m³]

    auto helmet = chrono_types::make_shared<ChBodyEasyEllipsoid>(
        helmet_axes,       // full axis lengths [m]
        effective_density, // density → correct mass
        true,              // create visualization asset
        false              // no collision geometry
    );
    helmet->SetName("helmet");
    helmet->SetPos(ChVector3d(0, 0, 0));
    helmet->SetFixed(false);
    // Override inertia with thin-shell ellipsoid approximation
    helmet->SetInertiaXX(ChVector3d(HELMET_IXX, HELMET_IYY, HELMET_IZZ));

    sys.AddBody(helmet);

    std::cout << "Helmet body:" << std::endl;
    std::cout << "  Mass    : " << helmet->GetMass() << " kg"  << std::endl;
    std::cout << "  Inertia : " << helmet->GetInertiaXX() << " kg·m²" << std::endl;

    // -------------------------------------------------------------------------
    // Apply Friedlander blast pressure as a time-varying force
    //
    // The blast source is located on the +Y side of the helmet.  The incident
    // overpressure wave travels in the −Y direction and exerts a net inward
    // (−Y) force on the +Y-facing masseter surface of the helmet.
    //
    // Masseter surface position (body frame): approximately (+Y lateral, −Z chin)
    //   y ≈ +0.12 m (lateral edge of ellipsoid)
    //   z ≈ −0.07 m (near mandible / chin region)
    // -------------------------------------------------------------------------
    const ChVector3d masseter_pos_body(0.0, 0.12, -0.07);  // body-frame [m]
    const ChVector3d blast_direction(0, -1, 0);             // −Y (inward)

    auto blast_force = chrono_types::make_shared<ChForce>();
    blast_force->SetMode(ChForce::ForceType::FORCE);
    // Force direction fixed in the world frame (blast comes from a fixed source)
    blast_force->SetAlign(ChForce::AlignmentFrame::WORLD_DIR);
    blast_force->SetMforce(1.0);  // dimensionless scale factor; actual force = 1.0 × modulation(t) [N]

    // Modulation function: F(t) = BlastForce(t) [N]
    auto blast_modulation = chrono_types::make_shared<ChFunctionLambda>();
    blast_modulation->SetFunction([td](double t) -> double { return BlastForce(t, td); });
    blast_force->SetModulation(blast_modulation);

    // AddForce must be called before SetFrame/SetVpoint/SetDir because those
    // methods call GetBody() internally to transform the coordinates.
    helmet->AddForce(blast_force);

    // Set application point (body frame) and force direction after the body is attached
    blast_force->SetFrame(ChForce::ReferenceFrame::BODY);
    blast_force->SetVrelpoint(masseter_pos_body);  // body-frame position (follows helmet)
    blast_force->SetDir(blast_direction);

    // -------------------------------------------------------------------------
    // Simulation loop
    // -------------------------------------------------------------------------
    std::cout << "\nRunning simulation..." << std::endl;
    std::cout << "  Time step : " << step_size * 1e6 << " μs" << std::endl;
    std::cout << "  End time  : " << t_end  * 1e3  << " ms"  << std::endl;

    // Output file columns:
    //   time[s]  pressure[Pa]  force[N]  pos_y[m]  vel_y[m/s]
    const std::string out_file = out_dir + "/helmet_blast_results.txt";
    std::ofstream ofile(out_file, std::ios::trunc);
    ofile << "# Helmet Blast Impact Simulation — Chrono SPH Equivalent Pressure Model\n";
    ofile << "# C4 charge : " << C4_MASS_KG << " kg at " << STANDOFF_M << " m standoff\n";
    ofile << "# TNT equiv : " << W_tnt << " kg,  Z = " << Z << " m/kg^(1/3)\n";
    ofile << "# Pso = "       << PSO_PA / 1e3 << " kPa,  td = " << td * 1e3 << " ms,  b = " << B_DECAY << "\n";
    ofile << "#\n";
    ofile << "# time[s]        pressure[Pa]    force[N]        pos_y[m]        vel_y[m/s]\n";

    double time        = 0.0;
    int    frame       = 0;
    // Write output at approximately every 1 μs (= every step for step_size = 1e-6)
    const int  out_interval = std::max(1, static_cast<int>(std::round(1.0e-6 / step_size)));

    while (time <= t_end) {
        if (frame % out_interval == 0) {
            const double P   = BlastPressure(time, td);
            const double F   = BlastForce(time, td);
            const double y   = helmet->GetPos().y();
            const double vy  = helmet->GetPosDt().y();
            ofile << std::scientific << std::setprecision(6)
                  << time << "  " << P << "  " << F << "  " << y << "  " << vy << "\n";
        }

        sys.DoStepDynamics(step_size);
        time += step_size;
        ++frame;
    }
    ofile.close();

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "  Output written to : " << out_file << std::endl;
    std::cout << "  Final position    : " << helmet->GetPos()   << " m"    << std::endl;
    std::cout << "  Final velocity    : " << helmet->GetPosDt() << " m/s"  << std::endl;
    std::cout << "  Final speed |v|   : " << helmet->GetPosDt().Length() << " m/s" << std::endl;

    return 0;
}
