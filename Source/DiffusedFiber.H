// SPDX-FileCopyrightText: 2023 - 2025 Yadong Zeng<zdsjtu@gmail.com> & ZhuXu Li<1246206018@qq.com>
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DIFFUSEDFIBER_H
#define DIFFUSEDFIBER_H

#include <DiffusedIB.H>

#if (AMREX_SPACEDIM == 2)
struct fiber_kernel{
    int id;                                    // 细丝编号
    Real amplitude{0.0};                      // 振幅A
    Real period{0.0};                         // 周期T
    Real length{0.0};                         // 长度L
    Real wave_number{0.0};                    // 波数n1
    Real phase{0.0};                          // 初始相位phi1
    int num_marker{0};                        // 拉格朗日点数

    Real y0_base{0.0};                             // 基线

    Vector<Real> x_marker;                    // 拉格朗日点x坐标数组
    Vector<Real> y_marker;                    // 拉格朗日点y坐标数组
    Vector<RealVect> velocity_marker;         // 拉格朗日点速度数组

    amrex::Gpu::DeviceVector<Real> x_marker_d;
    amrex::Gpu::DeviceVector<Real> y_marker_d;
    amrex::Gpu::DeviceVector<RealVect> velocity_marker_d;

    Real da{0.0};                          // 拉格朗日点面积
    RealVect ib_force{0.0,0.0};          // 浸没边界法计算得到的力
    int verbose{0};                           // 调试输出控制
};

class mFiberIter : public ParIter<0, 0, numAttri>{
public:
    using amrex::ParIter<0, 0, numAttri>::ParIter;
    using RealVector = amrex::ParIter<0, 0, numAttri>::ContainerType::RealVector;

    [[nodiscard]] const std::array<RealVector, numAttri>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    [[nodiscard]] const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    std::array<RealVector, numAttri>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

using mFiberContainer = ParticleContainer<0, 0, numAttri>;

class mFiber
{
public:
    explicit mFiber() = default;

    void InitFibers(const Vector<Real>& x0,
                     const Vector<Real>& y0,
                     const Vector<Real>& amplitude,
                     const Vector<Real>& period,
                     const Vector<Real>& length,
                     const Vector<Real>& wave_number,
                     const Vector<Real>& phase,
                     const Vector<int>& num_marker,
                     Real h,
                     int verbose = 0);

    void InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce,
                           Real dt = 0.1, DELTA_FUNCTION_TYPE type = FOUR_POINT_IB,
                           Real boundary_time = 0.0);

    void WriteFiberFile(int index);

    void InitialWithLargrangianPoints(const fiber_kernel& current_kernel);

    void ResetLargrangianPoints();

    void VelocityInterpolation(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    void ComputeLagrangianForce(Real dt, const fiber_kernel& kernel);

    void ForceSpreading(amrex::MultiFab &Euler, fiber_kernel& kernel, DELTA_FUNCTION_TYPE type);

    void VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const;

    void UpdateFibers(int iStep, Real time, Real dt);

    // Reconstruct prescribed positions and velocities, including on restart.
    void UpdateGeometry(Real time);

    static void WriteIBForceAndMoment(int step, amrex::Real time, amrex::Real dt, fiber_kernel& current_kernel);

    Vector<fiber_kernel> fiber_kernels;        // 所有细丝的fiber_kernel对象数组
    mFiberContainer *mContainer{nullptr};       // AMReX的ParticleContainer，存储拉格朗日点
    int max_largrangian_num = 0;               // 最大拉格朗日点数
    int verbose = 0;                           // 调试输出控制
    Real spend_time;
};

class Fibers{
public:
    static void create_fibers(const Geometry &gm,
                               const DistributionMapping & dm,
                               const BoxArray & ba);

    static mFiber* get_fibers();
    static void init_fiber(Real gravity, Real h);
    static void Initialize();
    static int FiberFinestLevel();

    inline static bool isInitial{false};
private:
    inline static mFiber* fiber = nullptr;
};

#endif
#endif
