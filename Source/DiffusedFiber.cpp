// SPDX-FileCopyrightText: 2023 - 2025 Yadong Zeng<zdsjtu@gmail.com> & ZhuXu Li<1246206018@qq.com>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <DiffusedFiber.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using namespace amrex;

#define LOCAL_LEVEL 0

#if (AMREX_SPACEDIM == 2)

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void deltaFunction(Real xf, Real xp, Real h, Real& value, DELTA_FUNCTION_TYPE type)
{
    Real rr = amrex::Math::abs(( xf - xp ) / h);

    switch (type) {
    case DELTA_FUNCTION_TYPE::FOUR_POINT_IB:
        if(rr >= 0 && rr < 1 ){
            value = 1.0 / 8.0 * ( 3.0 - 2.0 * rr + std::sqrt( 1.0 + 4.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else if (rr >= 1 && rr < 2) {
            value = 1.0 / 8.0 * ( 5.0 - 2.0 * rr - std::sqrt( -7.0 + 12.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    case DELTA_FUNCTION_TYPE::THREE_POINT_IB:
        if(rr >= 0.5 && rr < 1.5){
            value = 1.0 / 6.0 * ( 5.0 - 3.0 * rr - std::sqrt( - 3.0 * Math::powi<2>( 1 - rr) + 1.0 )) / h;
        }else if (rr >= 0 && rr < 0.5) {
            value = 1.0 / 3.0 * ( 1.0 + std::sqrt( 1.0 - 3 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    default:
        break;
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                    mParticle/mFiber member function                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
namespace FiberProperties{
    Vector<Real> x0{}, y0{};                    // 细丝起点坐标
    Vector<Real> amplitude{};                   // 振幅A
    Vector<Real> period{};                      // 周期T
    Vector<Real> length{};                      // 长度L
    Vector<Real> wave_number{};                 // 波数n1
    Vector<Real> phase{};                       // 初始相位phi1
    Vector<int> num_marker{};                   // 拉格朗日点数
    int euler_finest_level{0};                  // 最细网格等级
    int euler_velocity_index{0};                // 欧拉速度场索引
    int euler_force_index{0};                   // 欧拉力场索引
    Real euler_fluid_rho{0.0};                  // 流体密度
    int verbose{0};                             // 调试输出
    int loop_ns{2};                             // 欧拉-拉格朗日交互迭代次数
    int start_step{-1};                         // 开始步数
    int write_freq{1};                          // 输出频率
    Vector<Real> GLO, GHI;                      // 几何边界
    GpuArray<Real, 2> plo{0.0,0.0}, phi{0.0,0.0}, dx{0.0, 0.0};
}

void mFiber::InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce,
                              Real dt, DELTA_FUNCTION_TYPE type, Real boundary_time)
{
    if (verbose) amrex::Print() << "[Fiber] mFiber::InteractWithEuler\n";
    UpdateGeometry(boundary_time);

    // 清零时间记录
    spend_time = 0;
    auto InteractWithEulerStart = ParallelDescriptor::second();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt > 0.0, "fiber coupling requires dt > 0");

    MultiFab EulerForceTmp(EulerForce.boxArray(), EulerForce.DistributionMap(), 2, EulerForce.nGrow());


    // 清零所有细丝的IB力
    for(auto& fiber_kernel : fiber_kernels) {
        fiber_kernel.ib_force.scale(0.0);
    }

    // 欧拉-拉格朗日交互主循环
    int loop = FiberProperties::loop_ns;
    BL_ASSERT(loop > 0);
    while(loop > 0){
        if(verbose) amrex::Print() << "[Fiber] Ns loop index : " << loop << "\n";

        EulerForce.setVal(0.0);

        for(fiber_kernel& fiber_kernel : fiber_kernels){
            // 初始化拉格朗日点位置
            InitialWithLargrangianPoints(fiber_kernel);
            ResetLargrangianPoints();
            EulerForceTmp.setVal(0.0);

            // 保存旧的IB力
            auto ib_force = fiber_kernel.ib_force;
            fiber_kernel.ib_force.scale(0.0);

            // IBM核心步骤
            VelocityInterpolation(EulerVel, type);
            ComputeLagrangianForce(dt, fiber_kernel);
            ForceSpreading(EulerForceTmp, fiber_kernel, type);

            // 累加所有细丝的力到总欧拉力场
            MultiFab::Add(EulerForce, EulerForceTmp, 0, 0, 2, EulerForce.nGrow());

            // 恢复IB力
            fiber_kernel.ib_force += ib_force;
        }

        // 速度修正
        VelocityCorrection(EulerVel, EulerForce, dt);
        loop--;
    }

    spend_time += ParallelDescriptor::second() - InteractWithEulerStart;
}

void mFiber::InitFibers(const Vector<Real>& x0,
                         const Vector<Real>& y0,
                         const Vector<Real>& amplitude,
                         const Vector<Real>& period,
                         const Vector<Real>& length,
                         const Vector<Real>& wave_number,
                         const Vector<Real>& phase,
                         const Vector<int>& num_marker,
                         Real h,
                         int _verbose)
{
    verbose = _verbose;
    if (verbose) amrex::Print() << "[Fiber] mFiber::InitFibers\n";

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!x0.empty(), "at least one fiber is required");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        y0.size() == x0.size() && amplitude.size() == x0.size()
        && period.size() == x0.size() && length.size() == x0.size()
        && wave_number.size() == x0.size() && phase.size() == x0.size()
        && num_marker.size() == x0.size(), "fiber parameter arrays must have matching sizes");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(h > 0.0, "fiber grid spacing must be positive");

    for(int index = 0; index < x0.size(); index++){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(length[index] > 0.0 && period[index] > 0.0,
            "fiber length and period must be positive");
        fiber_kernel mKernel;
        mKernel.id = index + 1;
        mKernel.amplitude = amplitude[index];
        mKernel.period = period[index];
        mKernel.length = length[index];
        mKernel.wave_number = wave_number[index];
        mKernel.phase = phase[index];
        mKernel.num_marker = num_marker[index];

        mKernel.y0_base = y0[index];  //保存基线

        //计算拉格朗日点数，确保两端端点都有点，间距为网格长度h
        int num_points = static_cast<int>(std::ceil(length[index] / h)) + 1;
        // The legacy implementation reuses the first fiber's marker container.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(index == 0 || num_points == fiber_kernels[0].num_marker,
            "fibers sharing the marker container must have the same marker count");
        mKernel.num_marker = num_points;

        // 初始化拉格朗日点位置（使用计算得到的 mKernel.num_marker 保持一致）
        for (int i = 0; i < mKernel.num_marker; i++) {
            Real x_pos = x0[index] + i * h;
            Real y_pos = y0[index] + amplitude[index] *
                         sin(2 * Math::pi<Real>() * (-wave_number[index] * x_pos / length[index]) + phase[index]);

            mKernel.x_marker.push_back(x_pos);
            mKernel.y_marker.push_back(y_pos);
            mKernel.velocity_marker.push_back(RealVect(0.0, 0.0));
        }

        // 计算面积 da，仿照三维颗粒中 dv 的计算方式
        Real da = h * h;// 每个拉格朗日点对应的面积，假设为 h^2
        mKernel.da = da;

        //mKernel.x_marker_d.assign(mKernel.x_marker.begin(), mKernel.x_marker.end());
        //mKernel.y_marker_d.assign(mKernel.y_marker.begin(), mKernel.y_marker.end());
        //mKernel.velocity_marker_d.assign(mKernel.velocity_marker.begin(), mKernel.velocity_marker.end());
        // 同步到 device
        mKernel.x_marker_d.resize(mKernel.x_marker.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              mKernel.x_marker.begin(), mKernel.x_marker.end(),
                              mKernel.x_marker_d.begin());

        mKernel.y_marker_d.resize(mKernel.y_marker.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              mKernel.y_marker.begin(), mKernel.y_marker.end(),
                              mKernel.y_marker_d.begin());

        mKernel.velocity_marker_d.resize(mKernel.velocity_marker.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              mKernel.velocity_marker.begin(), mKernel.velocity_marker.end(),
                              mKernel.velocity_marker_d.begin());
        amrex::Gpu::streamSynchronize();

        fiber_kernels.emplace_back(mKernel);

        if (verbose) amrex::Print() << "Fiber " << index << ": x0=" << x0[index]
                                    << ", y0=" << y0[index] << ", amplitude=" << amplitude[index]
                                    << ", period=" << period[index] << ", length=" << length[index]
                                    << ", num_marker=" << num_marker[index] << "\n";
    }
}

void mFiber::InitialWithLargrangianPoints(const fiber_kernel& kernel)
{
    if (verbose) amrex::Print() << "mFiber::InitialWithLargrangianPoints\n";

    for(mFiberIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        if(np == 0) continue;
        auto *particles = pti.GetArrayOfStructs().data();

        // 获取细丝的拉格朗日点坐标
        //const auto* x_marker = kernel.x_marker.data();
        //const auto* y_marker = kernel.y_marker.data();
        const auto* x_marker = kernel.x_marker_d.dataPtr();
        const auto* y_marker = kernel.y_marker_d.dataPtr();

        amrex::ParallelFor( np, [=]
            AMREX_GPU_DEVICE (int i) noexcept {
                auto id = particles[i].id();
                // 设置拉格朗日点位置（沿细丝长度方向分布）
                particles[i].pos(0) = x_marker[id - 1];
                particles[i].pos(1) = y_marker[id - 1];
            }
        );
    }

    // 重新分布拉格朗日点
    mContainer->Redistribute();

    if (verbose) {
        amrex::Print() << "[fiber] : fiber marker num :" << mContainer->TotalNumberOfParticles() << "\n";
        mContainer->WriteAsciiFile(amrex::Concatenate("fiber", 1));
    }

    //if (verbose) {
        // 统计全局总数，而非单个进程本地数
    //    long local_cnt = static_cast<long>(mContainer->NumberOfParticles());
    //    ParallelDescriptor::ReduceLongSum(local_cnt);
    //    amrex::Print() << "[fiber] : fiber marker num (global) :" << local_cnt << "\n";
    //    mContainer->WriteAsciiFile(amrex::Concatenate("fiber", 1));
    //}
}

template <typename P = Particle<numAttri>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void VelocityInterpolation_cir(int p_iter, P const& p,
                               Real& Up, Real& Vp,
                               Array4<Real const> const& E, int EulerVIndex,
                               const int *lo, const int *hi,
                               GpuArray<Real, AMREX_SPACEDIM> const& plo,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               DELTA_FUNCTION_TYPE type)
{
    // 体积计算
    const Real d = dx[0] * dx[1];

    // 坐标计算
    const Real lx = (p.pos(0) - plo[0]) / dx[0];
    const Real ly = (p.pos(1) - plo[1]) / dx[1];
    int i = static_cast<int>(Math::floor(lx));
    int j = static_cast<int>(Math::floor(ly));


    // 初始化速度分量
    Up = 0; Vp = 0;


    // 循环
    for(int ii = -2; ii < 3; ii++){
        for(int jj = -2; jj < 3; jj++){
            Real tU, tV;
            const Real xi = plo[0] + (i + ii) * dx[0] + dx[0]/2;
            const Real yj = plo[1] + (j + jj) * dx[1] + dx[1]/2;
            deltaFunction(p.pos(0), xi, dx[0], tU, type);
            deltaFunction(p.pos(1), yj, dx[1], tV, type);
            const Real delta_value = tU * tV;
            Up += delta_value * E(i + ii, j + jj, 0, EulerVIndex    ) * d;
            Vp += delta_value * E(i + ii, j + jj, 0, EulerVIndex + 1) * d;
        }
    }
}

void mFiber::VelocityInterpolation(MultiFab &EulerVel, DELTA_FUNCTION_TYPE type)
{
    if (verbose) amrex::Print() << "\tmFiber::VelocityInterpolation\n";

    const auto& gm = mContainer->GetParGDB()->Geom(LOCAL_LEVEL);
    auto plo = gm.ProbLoArray();
    auto dx = gm.CellSizeArray();

    // 边界填充
    EulerVel.FillBoundary(FiberProperties::euler_velocity_index, 2, gm.periodicity());


    const int EulerVelocityIndex = FiberProperties::euler_velocity_index;

    for (mFiberIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti) {
        const Box& box = pti.validbox();
        auto& particles = pti.GetArrayOfStructs();
        auto *p_ptr = particles.data();
        const Long np = pti.numParticles();

        auto& attri = pti.GetAttribs();
        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        const auto& E = EulerVel.array(pti);

        amrex::ParallelFor(np, [=]
        AMREX_GPU_DEVICE (int i) noexcept {
            VelocityInterpolation_cir(i, p_ptr[i], Up[i], Vp[i], E, EulerVelocityIndex, box.loVect(), box.hiVect(), plo, dx, type);
        });
    }

    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("fiber", 2));
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ForceSpreading_cic (P const& p,
                         Real Px, Real Py,
                         ParticleReal& fxP, ParticleReal& fyP,
                         ParticleReal& mxP, ParticleReal& myP,
                         Array4<Real> const& E,
                         int EulerForceIndex,
                         Real da,
                         GpuArray<Real,AMREX_SPACEDIM> const& plo,
                         GpuArray<Real,AMREX_SPACEDIM> const& dx,
                         DELTA_FUNCTION_TYPE type)
{
    // 坐标计算
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    int i = static_cast<int>(Math::floor(lx));
    int j = static_cast<int>(Math::floor(ly));

    // 力乘以面积
    fxP *= da; fyP *= da;
    // 循环
    for(int ii = -2; ii < +3; ii++){
        for(int jj = -2; jj < +3; jj++){
            Real tU, tV;
            const Real xi = plo[0] + (i + ii) * dx[0] + dx[0]/2;
            const Real yj = plo[1] + (j + jj) * dx[1] + dx[1]/2;
            deltaFunction(p.pos(0), xi, dx[0], tU, type);
            deltaFunction(p.pos(1), yj, dx[1], tV, type);
            Real delta_value = tU * tV;
            Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, 0, EulerForceIndex  ), delta_value * fxP);
            Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, 0, EulerForceIndex+1), delta_value * fyP);
        }
    }
}

void mFiber::ForceSpreading(MultiFab & EulerForce, fiber_kernel& kernel, DELTA_FUNCTION_TYPE type)
{
    if (verbose) amrex::Print() << "\tmFiber::ForceSpreading\n";

    const auto& gm = mContainer->GetParGDB()->Geom(LOCAL_LEVEL);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.CellSizeArray();

    for(mFiberIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        const auto& particles = pti.GetArrayOfStructs();
        auto Uarray = EulerForce[pti].array();
        auto& attri = pti.GetAttribs();

        auto *const fxP_ptr = attri[P_ATTR::Fx_Marker].data();
        auto *const fyP_ptr = attri[P_ATTR::Fy_Marker].data();
        auto *const mxP_ptr = attri[P_ATTR::Mx_Marker].data();
        auto *const myP_ptr = attri[P_ATTR::My_Marker].data();
        const auto *const p_ptr = particles().data();

        // 细丝系统的中心坐标：使用起点作为参考点
        auto x0 = kernel.x_marker[0];  // 细丝起点 x 坐标
        auto y0 = kernel.y_marker[0];  // 细丝起点 y 坐标

        // 细丝系统的面积参数
        auto da = kernel.da;

        auto force_index = FiberProperties::euler_force_index;

        amrex::ParallelFor(np, [=]
        AMREX_GPU_DEVICE (int i) noexcept{
            ForceSpreading_cic(p_ptr[i], x0, y0,
                               fxP_ptr[i], fyP_ptr[i],
                               mxP_ptr[i], myP_ptr[i],
                               Uarray, force_index, da, plo, dxi, type);
        });
    }

    // 归约所有处理器的数据
    using pc = mFiberContainer::SuperParticleType;
    auto fx = amrex::ReduceSum(*mContainer, [=]AMREX_GPU_HOST_DEVICE(const pc& p)->ParticleReal{
        return p.rdata(P_ATTR::Fx_Marker);
    });
    auto fy = amrex::ReduceSum(*mContainer, [=]AMREX_GPU_HOST_DEVICE(const pc& p)->ParticleReal{
        return p.rdata(P_ATTR::Fy_Marker);
    });
    auto mx = amrex::ReduceSum(*mContainer, [=]AMREX_GPU_HOST_DEVICE(const pc& p)->ParticleReal{
        return p.rdata(P_ATTR::Mx_Marker);
    });
    auto my = amrex::ReduceSum(*mContainer, [=]AMREX_GPU_HOST_DEVICE(const pc& p)->ParticleReal{
        return p.rdata(P_ATTR::My_Marker);
    });

    // MPI归约
    amrex::ParallelAllReduce::Sum(fx, ParallelDescriptor::Communicator());
    amrex::ParallelAllReduce::Sum(fy, ParallelDescriptor::Communicator());
    amrex::ParallelAllReduce::Sum(mx, ParallelDescriptor::Communicator());
    amrex::ParallelAllReduce::Sum(my, ParallelDescriptor::Communicator());

    // 二维下只有x和y方向的力
    kernel.ib_force = {fx, fy};

    // 边界同步
    EulerForce.SumBoundary(FiberProperties::euler_force_index, 2, gm.periodicity());
}

void mFiber::ResetLargrangianPoints()
{
    if (verbose) amrex::Print() << "\tmFiber::ResetLargrangianPoints\n";

    for(mFiberIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        auto& attri = pti.GetAttribs();

        auto *const vUP_ptr = attri[P_ATTR::U_Marker].data();
        auto *const vVP_ptr = attri[P_ATTR::V_Marker].data();
        auto *const fxP_ptr = attri[P_ATTR::Fx_Marker].data();
        auto *const fyP_ptr = attri[P_ATTR::Fy_Marker].data();
        auto *const mxP_ptr = attri[P_ATTR::Mx_Marker].data();
        auto *const myP_ptr = attri[P_ATTR::My_Marker].data();

        amrex::ParallelFor(np, [=]
        AMREX_GPU_DEVICE (int i) noexcept{
            vUP_ptr[i] = 0.0;
            vVP_ptr[i] = 0.0;

            fxP_ptr[i] = 0.0;
            fyP_ptr[i] = 0.0;

            mxP_ptr[i] = 0.0;
            myP_ptr[i] = 0.0;

        });
    }
}

void mFiber::UpdateGeometry(Real time)
{
    for(auto& fiber_kernel : fiber_kernels){
        // 更新每个拉格朗日点的位置和速度
        for(int i = 0; i < fiber_kernel.num_marker; i++){
            Real x_pos = fiber_kernel.x_marker[i];
            Real y_pos = fiber_kernel.y0_base + fiber_kernel.amplitude *
                        sin(2 * Math::pi<Real>() * (time / fiber_kernel.period - fiber_kernel.wave_number * x_pos / fiber_kernel.length) + fiber_kernel.phase);

            // 更新位置
            fiber_kernel.y_marker[i] = y_pos;

            // 计算目标速度
            Real u_b_y = (2 * Math::pi<Real>() * fiber_kernel.amplitude / fiber_kernel.period) *
                        cos(2 * Math::pi<Real>() * (time / fiber_kernel.period - fiber_kernel.wave_number * x_pos / fiber_kernel.length) + fiber_kernel.phase);
            fiber_kernel.velocity_marker[i] = RealVect(0.0, u_b_y);
        }

        fiber_kernel.y_marker_d.resize(fiber_kernel.y_marker.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              fiber_kernel.y_marker.begin(), fiber_kernel.y_marker.end(),
                              fiber_kernel.y_marker_d.begin());

        fiber_kernel.velocity_marker_d.resize(fiber_kernel.velocity_marker.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              fiber_kernel.velocity_marker.begin(), fiber_kernel.velocity_marker.end(),
                              fiber_kernel.velocity_marker_d.begin());
        amrex::Gpu::streamSynchronize();
    }

}

void mFiber::UpdateFibers(int iStep, Real time, Real dt)
{
    UpdateGeometry(time);
    int fiber_write_freq = FiberProperties::write_freq;
    if (iStep % fiber_write_freq == 0) {
        for(auto& kernel: fiber_kernels)
            WriteIBForceAndMoment(iStep, time, dt, kernel);
    }
    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("fiber", 4));
}

void mFiber::ComputeLagrangianForce(Real dt, const fiber_kernel& kernel)
{
    if (verbose) amrex::Print() << "\tmFiber::ComputeLagrangianForce\n";

    for(mFiberIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        auto& attri = pti.GetAttribs();
        auto const* p_ptr = pti.GetArrayOfStructs().data();

        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        auto *FxP = attri[P_ATTR::Fx_Marker].data();
        auto *FyP = attri[P_ATTR::Fy_Marker].data();
        //const RealVect* velocity_marker_ptr = kernel.velocity_marker.data();
        const RealVect* velocity_marker_ptr = kernel.velocity_marker_d.data();

        // 系数（可改成输入参数）
        const Real k_pos = 1000.0;
        const Real k_vel = 1.0;
        //const Real k_penalty = 1000.0;

        //const Real* xex = kernel.x_marker.data();
        //const Real* xey = kernel.y_marker.data();
        const Real* xex = kernel.x_marker_d.data();
        const Real* xey = kernel.y_marker_d.data();
        amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
// IB力由速度差计算*********************************************************************************
            //const int id = p_ptr[i].id() - 1; // 粒子 id 从 1 开始，这里转换为 0-based 索引
            //Real u_b_x = 0.0;
            //Real u_b_y = fiber_kernel.velocity_marker[i][1]; // 从预计算的速度获取
            //Real u_b_y = velocity_marker_ptr[id][1]; // 从预计算的速度获取
            //计算IB力
            //FxP[i] = (u_b_x - Up[i]) / dt;
            //FyP[i] = (u_b_y - Vp[i]) / dt;

//由位置差计算IB力*********************************************************************************
            // X: 当前拉格朗日点位置
        //    const Real Xx = p_ptr[i].pos(0);
        //    const Real Xy = p_ptr[i].pos(1);

            // id 从 1 开始
        //    const int id = p_ptr[i].id() - 1;

            // Xe: 期望位置（由运动公式生成并保存在 kernel.x_marker/y_marker）
        //    const Real Xex = kernel.x_marker[id];
        //    const Real Xey = kernel.y_marker[id];

            // F = k (Xe - X)
        //    FxP[i] = k_penalty * (Xex - Xx);
        //    FyP[i] = k_penalty * (Xey - Xy);
//IB力由位置差和速度差共同决定***********************************************************************
            const int id = p_ptr[i].id() - 1; // 粒子 id 从 1 开始，这里转换为 0-based 索引

            // X（当前真实位置）与 Xe（期望位置）
            const Real Xx  = p_ptr[i].pos(0);
            const Real Xy  = p_ptr[i].pos(1);
            const Real Xex = xex[id];
            const Real Xey = xey[id];

            // U（插值得到的真实速度）与 Ue（期望速度）
            const Real Ux  = Up[i];
            const Real Uy  = Vp[i];
            const Real Uex = 0.0;
            const Real Uey = velocity_marker_ptr[id][1];

            // 合成惩罚力：F = k_pos (Xe - X) + k_vel (Ue - U) / dt
            FxP[i] = k_pos * (Xex - Xx) / dt + k_vel * (Uex - Ux) / dt;
            FyP[i] = k_pos * (Xey - Xy) / dt + k_vel * (Uey - Uy) / dt;
        });
    }
    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("fiber", 3));
}

void mFiber::VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const
{
    if(verbose) amrex::Print() << "\tmFiber::VelocityCorrection\n";

    // 速度修正
    MultiFab::Saxpy(Euler, dt, EulerForce, FiberProperties::euler_force_index, FiberProperties::euler_velocity_index, 2, 0);

}

void mFiber::WriteFiberFile(int index)
{
    mContainer->WriteAsciiFile(amrex::Concatenate("fiber", index));
}

void mFiber::WriteIBForceAndMoment(int step, amrex::Real time, amrex::Real dt, fiber_kernel& current_kernel)
{
    if(amrex::ParallelDescriptor::MyProc() != ParallelDescriptor::IOProcessorNumber()) return;

    std::string file("IB_Fiber_" + std::to_string(current_kernel.id) + ".csv");
    std::ofstream out_ib_force;

    std::string head;
    if(!fs::exists(file)){
        head = "iStep,time,Fx,Fy\n";
    }else{
        head = "";
    }

    out_ib_force.open(file, std::ios::app);
    if(!out_ib_force.is_open()){
        amrex::Print() << "[Fiber] write fiber file error , step: " << step;
    }else{
        out_ib_force << head << step << "," << time << ","
                     << current_kernel.ib_force[0] << "," << current_kernel.ib_force[1] << "\n";
    }
    out_ib_force.close();
}

void Fibers::create_fibers(const Geometry &gm,
                           const DistributionMapping &dm,
                           const BoxArray &ba)
{
    amrex::Print() << "[Fiber] : create Fiber Container\n";
    if (fiber->mContainer != nullptr) {
        delete fiber->mContainer;
        fiber->mContainer = nullptr;
    }
    fiber->mContainer = new mFiberContainer(gm, dm, ba);

    // 获取fiber tile
    std::pair<int, int> key{0, 0};
    auto& fiberTileTmp = fiber->mContainer->GetParticles(0)[key];

    // 插入fiber节点
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        for (int fiber_index = 0; fiber_index < fiber->fiber_kernels[0].num_marker; ++fiber_index) {
            mFiberContainer::ParticleType fiberNode;
            fiberNode.id() = fiber_index + 1;
            fiberNode.cpu() = ParallelDescriptor::MyProc();
            fiberNode.pos(0) = fiber->fiber_kernels[0].x_marker[fiber_index];
            fiberNode.pos(1) = fiber->fiber_kernels[0].y_marker[fiber_index];
            //fiberNode.pos(2) = 0.0; // 二维细丝，z坐标为0

            std::array<ParticleReal, numAttri> Fiber_attr{};
            Fiber_attr[U_Marker] = fiber->fiber_kernels[0].velocity_marker[fiber_index][0];
            Fiber_attr[V_Marker] = fiber->fiber_kernels[0].velocity_marker[fiber_index][1];
            Fiber_attr[W_Marker] = 0.0; // 二维细丝，W分量为0
            Fiber_attr[Fx_Marker] = 0.0;
            Fiber_attr[Fy_Marker] = 0.0;
            Fiber_attr[Fz_Marker] = 0.0; // 二维细丝，Fz分量为0

            fiberTileTmp.push_back(fiberNode);
            fiberTileTmp.push_back_real(Fiber_attr);
        }
    }
    fiber->mContainer->Redistribute();

    //FiberProperties::plo = gm.ProbLoArray();
    auto prob_lo = gm.ProbLoArray();
    FiberProperties::plo = amrex::GpuArray<Real,2>{prob_lo[0], prob_lo[1]};
    //FiberProperties::phi = gm.ProbHiArray();
    auto prob_hi = gm.ProbHiArray();
    FiberProperties::phi = amrex::GpuArray<Real,2>{prob_hi[0], prob_hi[1]};
    //FiberProperties::dx = gm.CellSizeArray();
    auto cell_size = gm.CellSizeArray();
    FiberProperties::dx = amrex::GpuArray<Real,2>{cell_size[0], cell_size[1]};
}
mFiber* Fibers::get_fibers()
{
    return fiber;
}

void Fibers::init_fiber(Real gravity, Real h)
{
    amrex::Print() << "[Fiber] : create Fiber's kernel\n";
    fiber = new mFiber;
    if (fiber != nullptr) {
        isInitial = true;
        fiber->InitFibers(
            FiberProperties::x0,
            FiberProperties::y0,
            FiberProperties::amplitude,
            FiberProperties::period,
            FiberProperties::length,
            FiberProperties::wave_number,
            FiberProperties::phase,
            FiberProperties::num_marker,
            h,
            FiberProperties::verbose);
    }
}

void Fibers::Initialize()
{
    ParmParse pp("fiber");

    std::string fiber_inputfile;
    pp.get("input", fiber_inputfile);

    if(!fiber_inputfile.empty()){
        ParmParse p_file(fiber_inputfile);
        p_file.getarr("x0",          FiberProperties::x0);
        p_file.getarr("y0",          FiberProperties::y0);
        p_file.getarr("amplitude",   FiberProperties::amplitude);
        p_file.getarr("period",      FiberProperties::period);
        p_file.getarr("length",      FiberProperties::length);
        p_file.getarr("wave_number", FiberProperties::wave_number);
        p_file.getarr("phase",       FiberProperties::phase);
        p_file.getarr("num_marker",  FiberProperties::num_marker);
        p_file.query("verbose",      FiberProperties::verbose);
        p_file.query("start_step",   FiberProperties::start_step);
        p_file.query("write_freq",   FiberProperties::write_freq);
        p_file.query("LOOP_NS",      FiberProperties::loop_ns);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(FiberProperties::loop_ns > 0
                                        && FiberProperties::write_freq > 0,
            "fiber LOOP_NS and write_freq must be positive");

        ParmParse ns("ns");
        ns.get("fluid_rho",          FiberProperties::euler_fluid_rho);

        ParmParse level_parse("amr");
        level_parse.get("max_level", FiberProperties::euler_finest_level);

        ParmParse geometry_parse("geometry");
        geometry_parse.getarr("prob_lo", FiberProperties::GLO);
        geometry_parse.getarr("prob_hi", FiberProperties::GHI);

        amrex::Print() << "[Fiber] : Reading fiber cfg file : " << fiber_inputfile << "\n"
                       << "             Fiber's level : " << FiberProperties::euler_finest_level << "\n";

    }else {
        amrex::Abort("[Fiber] : can't read fiber settings, pls check your config file \"fiber.input\"");
    }
}

int Fibers::FiberFinestLevel()
{
    return FiberProperties::euler_finest_level;
}

#endif
