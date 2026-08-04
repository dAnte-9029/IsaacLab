#include <Python.h>

#include <PxPhysicsAPI.h>

#include <carb/BindingsUtils.h>
#include <carb/Framework.h>
#include <carb/logging/Log.h>

#include <pxr/base/tf/token.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/schemaBase.h>
#include <pxr/usd/usd/typed.h>
#include <pxr/usd/usdPhysics/joint.h>

#include <omni/physx/IPhysxCustomJoint.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

CARB_BINDINGS("omni.flapping_bot.holonomic_constraint")

PXR_NAMESPACE_OPEN_SCOPE

class FlappingBotSchemaFlappingWingTrajectoryJoint final : public UsdPhysicsJoint
{
public:
    static const UsdSchemaKind schemaKind = UsdSchemaKind::ConcreteTyped;

    explicit FlappingBotSchemaFlappingWingTrajectoryJoint(const UsdPrim& prim = UsdPrim()) : UsdPhysicsJoint(prim)
    {
    }

    explicit FlappingBotSchemaFlappingWingTrajectoryJoint(const UsdSchemaBase& schemaObject) : UsdPhysicsJoint(schemaObject)
    {
    }

    ~FlappingBotSchemaFlappingWingTrajectoryJoint() override = default;

protected:
    UsdSchemaKind _GetSchemaKind() const override
    {
        return schemaKind;
    }

private:
    friend class UsdSchemaRegistry;

    static const TfType& _GetStaticTfType()
    {
        static TfType type = TfType::Find<FlappingBotSchemaFlappingWingTrajectoryJoint>();
        return type;
    }

    static bool _IsTypedSchema()
    {
        static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
        return isTyped;
    }

    const TfType& _GetTfType() const override
    {
        return _GetStaticTfType();
    }
};

TF_REGISTRY_FUNCTION(TfType)
{
    const TfType& type =
        TfType::Define<FlappingBotSchemaFlappingWingTrajectoryJoint, TfType::Bases<UsdPhysicsJoint>>();
    type.AddAlias(TfType::FindByName("UsdSchemaBase"), "FlappingWingTrajectoryJoint");
}

PXR_NAMESPACE_CLOSE_SCOPE

namespace
{

using omni::physx::CustomJointFlag;
using omni::physx::ICustomJointCallback;
using omni::physx::IPhysxCustomJoint;
using pxr::SdfPath;
using pxr::TfToken;
using namespace physx;

constexpr const char* kJointTypeName = "FlappingWingTrajectoryJoint";

struct TrajectoryTarget
{
    PxReal positionRad{ 0.0f };
    PxReal velocityRadS{ 0.0f };
};

struct TrajectoryConstraintData
{
    PxTransform constraintToBody[2]{ PxTransform(PxIdentity), PxTransform(PxIdentity) };
    PxReal targetPositionRad{ 0.0f };
    PxReal targetVelocityRadS{ 0.0f };
    PxReal maxImpulseNs{ PX_MAX_F32 };
};

struct JointState
{
    PxRigidBody* bodies[2]{ nullptr, nullptr };
    PxTransform localFrames[2]{ PxTransform(PxIdentity), PxTransform(PxIdentity) };
    TrajectoryConstraintData data;
};

std::mutex gMutex;
std::unordered_map<std::string, std::unique_ptr<JointState>> gJointStates;
std::unordered_map<std::string, TrajectoryTarget> gTargets;
IPhysxCustomJoint* gCustomJoint = nullptr;
size_t gRegistrationId = omni::physx::kInvalidCustomJointRegId;
std::atomic<PxU64> gSolverPrepCount{ 0 };
std::atomic<PxU64> gReleaseEnterCount{ 0 };
std::atomic<PxU64> gReleaseLockCount{ 0 };
std::atomic<PxU64> gReleaseExitCount{ 0 };
std::atomic<PxReal> gLastActualPositionRad{ 0.0f };
std::atomic<PxReal> gLastTargetPositionRad{ 0.0f };
std::atomic<PxReal> gLastPositionErrorRad{ 0.0f };

PxReal wrapAngle(PxReal angle)
{
    constexpr PxReal twoPi = PxReal(2.0 * 3.14159265358979323846);
    constexpr PxReal pi = PxReal(3.14159265358979323846);
    angle = std::fmod(angle + pi, twoPi);
    if (angle < 0.0f)
    {
        angle += twoPi;
    }
    return angle - pi;
}

PxReal extractTwistAngleX(const PxQuat& relativeOrientation)
{
    PxReal w = relativeOrientation.w;
    PxReal x = relativeOrientation.x;
    if (w < 0.0f)
    {
        w = -w;
        x = -x;
    }
    const PxReal norm = std::sqrt(w * w + x * x);
    if (norm <= PxReal(1.0e-12f))
    {
        return 0.0f;
    }
    return wrapAngle(PxReal(2.0f) * std::atan2(x / norm, w / norm));
}

PxU32 solverPrep(
    Px1DConstraint* constraints,
    PxVec3p& body0WorldOffset,
    PxU32 maxConstraints,
    PxConstraintInvMassScale& invMassScale,
    const void* constantBlock,
    const PxTransform& body0ToWorld,
    const PxTransform& body1ToWorld,
    bool useExtendedLimits,
    PxVec3p& constraint0ToWorldOut,
    PxVec3p& constraint1ToWorldOut)
{
    PX_UNUSED(useExtendedLimits);
    if (constraints == nullptr || constantBlock == nullptr || maxConstraints == 0)
    {
        return 0;
    }

    const auto& data = *static_cast<const TrajectoryConstraintData*>(constantBlock);
    const PxTransform constraint0ToWorld = body0ToWorld.transform(data.constraintToBody[0]);
    const PxTransform constraint1ToWorld = body1ToWorld.transform(data.constraintToBody[1]);

    constraint0ToWorldOut = constraint0ToWorld.p;
    constraint1ToWorldOut = constraint1ToWorld.p;
    body0WorldOffset = constraint1ToWorld.p - body0ToWorld.p;
    invMassScale = PxConstraintInvMassScale(1.0f, 1.0f, 1.0f, 1.0f);

    const PxQuat relativeOrientation = constraint0ToWorld.q.getConjugate() * constraint1ToWorld.q;
    const PxReal actualPositionRad = extractTwistAngleX(relativeOrientation);
    const PxReal positionErrorRad = wrapAngle(actualPositionRad - data.targetPositionRad);
    const PxVec3 hingeAxisWorld = constraint0ToWorld.q.getBasisVector0().getNormalized();
    gSolverPrepCount.fetch_add(1, std::memory_order_relaxed);
    gLastActualPositionRad.store(actualPositionRad, std::memory_order_relaxed);
    gLastTargetPositionRad.store(data.targetPositionRad, std::memory_order_relaxed);
    gLastPositionErrorRad.store(positionErrorRad, std::memory_order_relaxed);

    Px1DConstraint& row = constraints[0];
    row = Px1DConstraint{};
    row.linear0 = PxVec3(0.0f);
    row.angular0 = -hingeAxisWorld;
    row.linear1 = PxVec3(0.0f);
    row.angular1 = -hingeAxisWorld;
    row.geometricError = positionErrorRad;
    row.velocityTarget = data.targetVelocityRadS;
    row.minImpulse = -data.maxImpulseNs;
    row.maxImpulse = data.maxImpulseNs;
    row.flags = Px1DConstraintFlag::eOUTPUT_FORCE | Px1DConstraintFlag::eANGULAR_CONSTRAINT;
    row.solveHint = PxConstraintSolveHint::eEQUALITY;
    row.mods.bounce.restitution = 0.0f;
    row.mods.bounce.velocityThreshold = 0.0f;
    return 1;
}

JointState* findJointStateLocked(const SdfPath& path)
{
    const auto found = gJointStates.find(path.GetString());
    return found == gJointStates.end() ? nullptr : found->second.get();
}

bool createJoint(
    SdfPath path,
    long stageId,
    PxRigidActor* actor0,
    const PxTransform& localFrame0,
    PxRigidActor* actor1,
    const PxTransform& localFrame1,
    CustomJointFlag::Enum& constraintFlags,
    void* userData)
{
    PX_UNUSED(stageId);
    PX_UNUSED(userData);
    if (actor0 == nullptr || actor1 == nullptr)
    {
        CARB_LOG_ERROR("%s requires two rigid actors at %s.", kJointTypeName, path.GetText());
        return false;
    }

    PxRigidBody* body0 = actor0->is<PxRigidBody>();
    PxRigidBody* body1 = actor1->is<PxRigidBody>();
    if (body0 == nullptr || body1 == nullptr)
    {
        CARB_LOG_ERROR("%s requires two PxRigidBody actors at %s.", kJointTypeName, path.GetText());
        return false;
    }

    auto state = std::make_unique<JointState>();
    state->bodies[0] = body0;
    state->bodies[1] = body1;
    state->localFrames[0] = localFrame0.getNormalized();
    state->localFrames[1] = localFrame1.getNormalized();
    state->data.constraintToBody[0] = body0->getCMassLocalPose().transformInv(state->localFrames[0]);
    state->data.constraintToBody[1] = body1->getCMassLocalPose().transformInv(state->localFrames[1]);

    const std::string pathString = path.GetString();
    std::lock_guard<std::mutex> guard(gMutex);
    if (const auto target = gTargets.find(pathString); target != gTargets.end())
    {
        state->data.targetPositionRad = target->second.positionRad;
        state->data.targetVelocityRadS = target->second.velocityRadS;
    }
    gJointStates[pathString] = std::move(state);
    constraintFlags = CustomJointFlag::eALWAYS_UPDATE;
    return true;
}

void releaseJoint(SdfPath path, void* userData)
{
    PX_UNUSED(userData);
    gReleaseEnterCount.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(gMutex);
    gReleaseLockCount.fetch_add(1, std::memory_order_relaxed);
    gJointStates.erase(path.GetString());
    gReleaseExitCount.fetch_add(1, std::memory_order_relaxed);
}

void* prepareJointData(SdfPath path, void* userData)
{
    PX_UNUSED(userData);
    std::lock_guard<std::mutex> guard(gMutex);
    JointState* state = findJointStateLocked(path);
    return state == nullptr ? nullptr : &state->data;
}

void onComShift(SdfPath path, PxU32 actorIndex, void* userData)
{
    PX_UNUSED(userData);
    if (actorIndex > 1)
    {
        return;
    }
    std::lock_guard<std::mutex> guard(gMutex);
    JointState* state = findJointStateLocked(path);
    if (state != nullptr && state->bodies[actorIndex] != nullptr)
    {
        state->data.constraintToBody[actorIndex] =
            state->bodies[actorIndex]->getCMassLocalPose().transformInv(state->localFrames[actorIndex]);
    }
}

void onOriginShift(SdfPath path, const PxVec3& shift, void* userData)
{
    PX_UNUSED(path);
    PX_UNUSED(shift);
    PX_UNUSED(userData);
}

const void* getConstantBlock(SdfPath path, void* userData)
{
    PX_UNUSED(userData);
    std::lock_guard<std::mutex> guard(gMutex);
    JointState* state = findJointStateLocked(path);
    return state == nullptr ? nullptr : &state->data;
}

bool registerSchemaType()
{
    const pxr::TfType type = pxr::UsdSchemaRegistry::GetTypeFromSchemaTypeName(pxr::TfToken(kJointTypeName));
    const pxr::TfType jointType = pxr::TfType::FindByName("UsdPhysicsJoint");
    return !type.IsUnknown() && !jointType.IsUnknown() && type.IsA(jointType);
}

bool initializeNative()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gRegistrationId != omni::physx::kInvalidCustomJointRegId)
    {
        return true;
    }
    if (!registerSchemaType())
    {
        CARB_LOG_ERROR("Failed to register USD type %s.", kJointTypeName);
        return false;
    }
    carb::Framework* framework = carb::getFramework();
    if (framework == nullptr)
    {
        CARB_LOG_ERROR("Carbonite framework is unavailable.");
        return false;
    }
    gCustomJoint = framework->acquireInterface<IPhysxCustomJoint>();
    if (gCustomJoint == nullptr)
    {
        CARB_LOG_ERROR("IPhysxCustomJoint v1.0 is unavailable.");
        return false;
    }

    ICustomJointCallback callbacks;
    callbacks.createJointFn = createJoint;
    callbacks.releaseJointFn = releaseJoint;
    callbacks.prepareJointDataFn = prepareJointData;
    callbacks.onComShiftFn = onComShift;
    callbacks.onOriginShift = onOriginShift;
    callbacks.getConstantBlockFn = getConstantBlock;
    callbacks.userData = nullptr;
    gRegistrationId = gCustomJoint->registerCustomJoint(
        TfToken(kJointTypeName), callbacks, solverPrep, sizeof(TrajectoryConstraintData));
    if (gRegistrationId == omni::physx::kInvalidCustomJointRegId)
    {
        framework->releaseInterface(gCustomJoint);
        gCustomJoint = nullptr;
        CARB_LOG_ERROR("PhysX rejected custom joint type %s.", kJointTypeName);
        return false;
    }
    CARB_LOG_INFO("Registered %s with IPhysxCustomJoint v1.0.", kJointTypeName);
    return true;
}

void shutdownNative()
{
    IPhysxCustomJoint* customJoint = nullptr;
    size_t registrationId = omni::physx::kInvalidCustomJointRegId;
    {
        std::lock_guard<std::mutex> guard(gMutex);
        customJoint = gCustomJoint;
        registrationId = gRegistrationId;
        gCustomJoint = nullptr;
        gRegistrationId = omni::physx::kInvalidCustomJointRegId;
        gJointStates.clear();
        gTargets.clear();
    }
    if (customJoint != nullptr)
    {
        if (registrationId != omni::physx::kInvalidCustomJointRegId)
        {
            customJoint->unregisterCustomJoint(registrationId);
        }
        if (carb::Framework* framework = carb::getFramework(); framework != nullptr)
        {
            framework->releaseInterface(customJoint);
        }
    }
}

bool parseStringSequence(PyObject* object, std::vector<std::string>& values, const char* name)
{
    PyObject* sequence = PySequence_Fast(object, name);
    if (sequence == nullptr)
    {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    values.reserve(static_cast<size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < size; ++index)
    {
        const char* value = PyUnicode_AsUTF8(items[index]);
        if (value == nullptr)
        {
            Py_DECREF(sequence);
            return false;
        }
        values.emplace_back(value);
    }
    Py_DECREF(sequence);
    return true;
}

bool parseFloatSequence(PyObject* object, std::vector<PxReal>& values, const char* name)
{
    PyObject* sequence = PySequence_Fast(object, name);
    if (sequence == nullptr)
    {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    values.reserve(static_cast<size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < size; ++index)
    {
        const double value = PyFloat_AsDouble(items[index]);
        if (PyErr_Occurred() != nullptr || !std::isfinite(value))
        {
            if (PyErr_Occurred() == nullptr)
            {
                PyErr_Format(PyExc_ValueError, "%s must contain only finite values.", name);
            }
            Py_DECREF(sequence);
            return false;
        }
        values.push_back(static_cast<PxReal>(value));
    }
    Py_DECREF(sequence);
    return true;
}

PyObject* pyInitialize(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    if (initializeNative())
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* pyShutdown(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    shutdownNative();
    Py_RETURN_NONE;
}

PyObject* pySetTargets(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PyObject* pathObject = nullptr;
    PyObject* positionObject = nullptr;
    PyObject* velocityObject = nullptr;
    if (!PyArg_ParseTuple(args, "OOO:set_targets", &pathObject, &positionObject, &velocityObject))
    {
        return nullptr;
    }

    std::vector<std::string> paths;
    std::vector<PxReal> positions;
    std::vector<PxReal> velocities;
    if (!parseStringSequence(pathObject, paths, "paths must be a sequence of strings") ||
        !parseFloatSequence(positionObject, positions, "positions") ||
        !parseFloatSequence(velocityObject, velocities, "velocities"))
    {
        return nullptr;
    }
    if (paths.size() != positions.size() || paths.size() != velocities.size())
    {
        PyErr_SetString(PyExc_ValueError, "paths, positions and velocities must have equal length.");
        return nullptr;
    }

    std::vector<SdfPath> dirtyPaths;
    dirtyPaths.reserve(paths.size());
    {
        std::lock_guard<std::mutex> guard(gMutex);
        if (gRegistrationId == omni::physx::kInvalidCustomJointRegId || gCustomJoint == nullptr)
        {
            PyErr_SetString(PyExc_RuntimeError, "The native custom joint is not registered.");
            return nullptr;
        }
        for (size_t index = 0; index < paths.size(); ++index)
        {
            const SdfPath path(paths[index]);
            if (!path.IsAbsolutePath() || !path.IsPrimPath())
            {
                PyErr_Format(PyExc_ValueError, "Invalid absolute prim path: %s", paths[index].c_str());
                return nullptr;
            }
            gTargets[paths[index]] = TrajectoryTarget{ positions[index], velocities[index] };
            if (const auto found = gJointStates.find(paths[index]); found != gJointStates.end())
            {
                found->second->data.targetPositionRad = positions[index];
                found->second->data.targetVelocityRadS = velocities[index];
            }
            dirtyPaths.push_back(path);
        }
    }
    for (const SdfPath& path : dirtyPaths)
    {
        gCustomJoint->markJointDirty(path);
    }
    Py_RETURN_NONE;
}

PyObject* pyGetActiveJointCount(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    std::lock_guard<std::mutex> guard(gMutex);
    return PyLong_FromSize_t(gJointStates.size());
}

PyObject* pyGetRegistrationId(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    std::lock_guard<std::mutex> guard(gMutex);
    return PyLong_FromSize_t(gRegistrationId);
}

PyObject* pyGetJointTypeName(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    return PyUnicode_FromString(kJointTypeName);
}

PyObject* pyGetDebugState(PyObject* self, PyObject* args)
{
    PX_UNUSED(self);
    PX_UNUSED(args);
    return Py_BuildValue(
        "{s:K,s:d,s:d,s:d,s:K,s:K,s:K}",
        "solver_prep_count",
        static_cast<unsigned long long>(gSolverPrepCount.load(std::memory_order_relaxed)),
        "actual_position_rad",
        static_cast<double>(gLastActualPositionRad.load(std::memory_order_relaxed)),
        "target_position_rad",
        static_cast<double>(gLastTargetPositionRad.load(std::memory_order_relaxed)),
        "position_error_rad",
        static_cast<double>(gLastPositionErrorRad.load(std::memory_order_relaxed)),
        "release_enter_count",
        static_cast<unsigned long long>(gReleaseEnterCount.load(std::memory_order_relaxed)),
        "release_lock_count",
        static_cast<unsigned long long>(gReleaseLockCount.load(std::memory_order_relaxed)),
        "release_exit_count",
        static_cast<unsigned long long>(gReleaseExitCount.load(std::memory_order_relaxed)));
}

PyMethodDef kMethods[] = {
    { "initialize", pyInitialize, METH_NOARGS, "Register the PhysX custom joint type." },
    { "shutdown", pyShutdown, METH_NOARGS, "Unregister the PhysX custom joint type." },
    { "set_targets", pySetTargets, METH_VARARGS, "Set batched joint position and velocity targets." },
    { "get_active_joint_count", pyGetActiveJointCount, METH_NOARGS, "Return the number of live custom joints." },
    { "get_registration_id", pyGetRegistrationId, METH_NOARGS, "Return the PhysX registration id." },
    { "get_joint_type_name", pyGetJointTypeName, METH_NOARGS, "Return the registered USD joint type name." },
    { "get_debug_state", pyGetDebugState, METH_NOARGS, "Return the latest solver-prep diagnostics." },
    { nullptr, nullptr, 0, nullptr },
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native PhysX holonomic trajectory constraint.",
    -1,
    kMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__native()
{
    return PyModule_Create(&kModule);
}
