// NVIDIA CUPTI resource callbacks only: no performance counters or driver changes.
#include <cupti.h>
#include <fstream>
#include <mutex>
#include <string>

static CUpti_SubscriberHandle subscriber;
static std::string directory;
static std::mutex guard;
static size_t bytes = 0;
static void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *data) {
    if (domain != CUPTI_CB_DOMAIN_RESOURCE || id != CUPTI_CBID_RESOURCE_MODULE_LOADED) return;
    const auto *resource = static_cast<const CUpti_ResourceData *>(data);
    const auto *module = static_cast<const CUpti_ModuleResourceData *>(resource->resourceDescriptor);
    if (!module || !module->pCubin || !module->cubinSize) return;
    std::lock_guard<std::mutex> lock(guard);
    const std::string name = "module_" + std::to_string(module->moduleId) + ".cubin";
    std::ofstream log(directory + "/modules.tsv", std::ios::app);
    if (bytes + module->cubinSize > 1024ull*1024*1024) {
        log << module->moduleId << '\t' << module->cubinSize << "\tskipped_1GiB_cap\n";
        return;
    }
    std::ofstream output(directory + "/" + name, std::ios::binary);
    output.write(module->pCubin, module->cubinSize);
    log << module->moduleId << '\t' << module->cubinSize << '\t' << name << '\n';
    bytes += module->cubinSize;
}
extern "C" int start_capture(const char *path) {
    directory = path;
    auto result = cuptiSubscribe(&subscriber, callback, nullptr);
    if (result != CUPTI_SUCCESS) return result;
    return cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
}
extern "C" int stop_capture() { return cuptiUnsubscribe(subscriber); }
