use std::ffi::{CStr, c_void};
use std::ptr::{self, NonNull};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use objc2::AnyThread;
use objc2_app_kit::NSWorkspace;
use objc2_core_audio::{
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, CATapDescription, CATapMuteBehavior, kAudioAggregateDeviceNameKey,
    kAudioAggregateDeviceTapAutoStartKey, kAudioAggregateDeviceTapListKey,
    kAudioAggregateDeviceUIDKey, kAudioEndPointDeviceIsPrivateKey,
    kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, kAudioProcessPropertyBundleID,
    kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
};
use objc2_core_foundation::{
    CFArray, CFDictionary, CFMutableDictionary, CFRetained, CFString, kCFAllocatorDefault,
    kCFTypeArrayCallBacks, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks,
};
use objc2_foundation::{NSArray, NSNumber, NSString};

use super::TappableApp;

/// Private Core Audio tap + aggregate. Destroyed on drop, after the cpal stream.
pub struct ProcessTap {
    pub uid: String,
    tap_id: AudioObjectID,
    aggregate_id: AudioObjectID,
}

impl Drop for ProcessTap {
    fn drop(&mut self) {
        unsafe {
            let _ = AudioHardwareDestroyAggregateDevice(self.aggregate_id);
            let _ = AudioHardwareDestroyProcessTap(self.tap_id);
        }
    }
}

/// Create a process-scoped tap for `query` (app name or bundle ID).
pub fn create_app_tap(query: &str) -> Result<ProcessTap> {
    let resolved = resolve_app(query)?;
    let process_ids = matching_process_objects(&resolved.bundle_id);
    if process_ids.is_empty() {
        anyhow::bail!(
            "found {} ({}) but it has no Core Audio process objects yet; start playback in the app and retry",
            resolved.name,
            resolved.bundle_id
        );
    }

    eprintln!(
        "Capturing app {} ({}) with {} audio process object(s).",
        resolved.name,
        resolved.bundle_id,
        process_ids.len()
    );
    eprintln!(
        "This includes every tab/window of that app. If you hear playback but get silence, grant System Audio Recording to Terminal/Cursor in System Settings → Privacy & Security."
    );

    let numbers: Vec<_> = process_ids
        .iter()
        .map(|id| NSNumber::initWithUnsignedInt(NSNumber::alloc(), *id))
        .collect();
    let processes = NSArray::from_retained_slice(&numbers);
    let bundle_id = NSString::from_str(&resolved.bundle_id);
    let bundle_ids = NSArray::from_retained_slice(std::slice::from_ref(&bundle_id));
    let output_uid = default_output_uid()?;

    let tap_desc = unsafe {
        CATapDescription::initWithProcesses_andDeviceUID_withStream(
            CATapDescription::alloc(),
            &processes,
            &output_uid,
            0,
        )
    };
    unsafe {
        tap_desc.setMuteBehavior(CATapMuteBehavior::Unmuted);
        tap_desc.setName(&NSString::from_str(&format!(
            "konnnyaku {}",
            resolved.bundle_id
        )));
        tap_desc.setPrivate(true);
        // This initializer's process list is an exclusion list when exclusive is true.
        tap_desc.setExclusive(false);
        tap_desc.setBundleIDs(&bundle_ids);
        tap_desc.setProcessRestoreEnabled(true);
    }

    let mut tap_id = 0;
    let status = unsafe { AudioHardwareCreateProcessTap(Some(tap_desc.as_ref()), &mut tap_id) };
    check_os_status(status).context("create process tap")?;
    let tap_uid = unsafe { tap_desc.UUID().UUIDString() };

    let pid = std::process::id();
    let agg_uid = format!("com.konnnyaku.AppTap.{pid}.{tap_id}");
    let agg_name = format!("konnnyaku {}", resolved.name);
    let properties = aggregate_properties(&tap_uid, &agg_uid, &agg_name);
    let mut aggregate_id: AudioObjectID = 0;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(properties.as_ref(), NonNull::from(&mut aggregate_id))
    };
    check_os_status(status).context("create tap aggregate device")?;

    Ok(ProcessTap {
        uid: agg_uid,
        tap_id,
        aggregate_id,
    })
}

/// Running apps that currently have Core Audio process objects.
pub fn list_tappable_apps() -> Result<Vec<TappableApp>> {
    let processes = process_objects().context("list Core Audio process objects")?;
    let mut apps = Vec::new();
    for app in NSWorkspace::sharedWorkspace().runningApplications() {
        let Some(bundle_id) = app.bundleIdentifier().map(|id| id.to_string()) else {
            continue;
        };
        if bundle_id.is_empty() {
            continue;
        }
        let process_count = processes
            .iter()
            .filter(|process| bundle_matches(&process.bundle_id, &bundle_id))
            .count();
        if process_count == 0 {
            continue;
        }
        let name = app
            .localizedName()
            .map(|name| name.to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| bundle_id.clone());
        apps.push(TappableApp {
            name,
            bundle_id,
            process_count,
        });
    }
    collapse_helper_apps(&mut apps);
    apps.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
    Ok(apps)
}

fn collapse_helper_apps(apps: &mut Vec<TappableApp>) {
    apps.sort_by(|left, right| left.bundle_id.len().cmp(&right.bundle_id.len()));
    let mut kept: Vec<TappableApp> = Vec::new();
    for app in apps.drain(..) {
        let is_helper = kept
            .iter()
            .any(|parent| bundle_matches(&app.bundle_id, &parent.bundle_id));
        if is_helper {
            continue;
        }
        if let Some(existing) = kept
            .iter_mut()
            .find(|parent| parent.bundle_id.eq_ignore_ascii_case(&app.bundle_id))
        {
            if app.process_count > existing.process_count {
                existing.process_count = app.process_count;
            }
            continue;
        }
        kept.push(app);
    }
    *apps = kept;
}

struct ResolvedApp {
    name: String,
    bundle_id: String,
}

struct AudioProcess {
    object_id: AudioObjectID,
    bundle_id: String,
}

fn resolve_app(query: &str) -> Result<ResolvedApp> {
    let query = query.trim();
    if query.is_empty() {
        anyhow::bail!("--app requires an application name or bundle ID");
    }

    let running: Vec<ResolvedApp> = NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .filter_map(|app| {
            let bundle_id = app.bundleIdentifier()?.to_string();
            if bundle_id.is_empty() {
                return None;
            }
            let name = app
                .localizedName()
                .map(|name| name.to_string())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| bundle_id.clone());
            Some(ResolvedApp { name, bundle_id })
        })
        .collect();

    let query_lower = query.to_lowercase();
    let looks_like_bundle_id = query.contains('.');

    let mut matches: Vec<ResolvedApp> = running
        .into_iter()
        .filter(|app| {
            if looks_like_bundle_id {
                app.bundle_id.eq_ignore_ascii_case(query)
            } else {
                app.name.eq_ignore_ascii_case(query)
            }
        })
        .collect();

    if matches.is_empty() && !looks_like_bundle_id {
        matches = NSWorkspace::sharedWorkspace()
            .runningApplications()
            .iter()
            .filter_map(|app| {
                let bundle_id = app.bundleIdentifier()?.to_string();
                let name = app.localizedName()?.to_string();
                if name.to_lowercase().contains(&query_lower) {
                    Some(ResolvedApp { name, bundle_id })
                } else {
                    None
                }
            })
            .collect();
    }

    dedupe_bundle_ids(&mut matches);

    match matches.len() {
        0 => Err(anyhow!(
            "no running application matched '{query}'. Use `konnnyaku apps` to list tappable apps"
        )),
        1 => Ok(matches.remove(0)),
        _ => {
            let listed = matches
                .iter()
                .map(|app| format!("{} ({})", app.name, app.bundle_id))
                .collect::<Vec<_>>()
                .join(", ");
            Err(anyhow!(
                "multiple applications matched '{query}': {listed}. Pass a bundle ID"
            ))
        }
    }
}

fn matching_process_objects(bundle_id: &str) -> Vec<AudioObjectID> {
    process_objects()
        .unwrap_or_default()
        .into_iter()
        .filter(|process| bundle_matches(&process.bundle_id, bundle_id))
        .map(|process| process.object_id)
        .collect()
}

fn bundle_matches(candidate: &str, target: &str) -> bool {
    candidate.eq_ignore_ascii_case(target)
        || candidate
            .to_lowercase()
            .starts_with(&format!("{}.", target.to_lowercase()))
}

fn dedupe_bundle_ids(apps: &mut Vec<ResolvedApp>) {
    let mut seen = Vec::new();
    apps.retain(|app| {
        let id = app.bundle_id.to_lowercase();
        if seen.contains(&id) {
            false
        } else {
            seen.push(id);
            true
        }
    });
}

fn process_objects() -> Result<Vec<AudioProcess>> {
    let address = global_address(kAudioHardwarePropertyProcessObjectList);
    let mut size = 0_u32;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
        )
    };
    check_os_status(status).context("query process object list size")?;
    if size == 0 {
        return Ok(Vec::new());
    }

    let count = size as usize / size_of::<AudioObjectID>();
    let mut ids = vec![0_u32; count];
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::new(ids.as_mut_ptr())
                .expect("process id buffer")
                .cast(),
        )
    };
    check_os_status(status).context("read process object list")?;

    let mut processes = Vec::new();
    for object_id in ids {
        if object_id == 0 {
            continue;
        }
        if let Some(bundle_id) = process_bundle_id(object_id) {
            processes.push(AudioProcess {
                object_id,
                bundle_id,
            });
        }
    }
    Ok(processes)
}

fn process_bundle_id(object_id: AudioObjectID) -> Option<String> {
    let address = global_address(kAudioProcessPropertyBundleID);
    let mut cfstring: *mut CFString = ptr::null_mut();
    let mut size = size_of::<*mut CFString>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut cfstring).cast(),
        )
    };
    if status != 0 || cfstring.is_null() {
        return None;
    }
    let retained = unsafe { objc2::rc::Retained::<NSString>::retain(cfstring.cast())? };
    let bundle_id = retained.to_string();
    if bundle_id.is_empty() {
        None
    } else {
        Some(bundle_id)
    }
}

fn aggregate_properties(
    tap_uid: &NSString,
    agg_uid: &str,
    agg_name: &str,
) -> CFRetained<CFDictionary> {
    unsafe {
        let tap_inner = CFMutableDictionary::new(
            kCFAllocatorDefault,
            2,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("tap dictionary");
        CFMutableDictionary::set_value(
            Some(tap_inner.as_ref()),
            &*cf_key(kAudioSubTapUIDKey) as *const _ as *const c_void,
            tap_uid as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(tap_inner.as_ref()),
            &*cf_key(kAudioSubTapDriftCompensationKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );

        let taps_list = [tap_inner];
        let taps = CFArray::new(
            kCFAllocatorDefault,
            taps_list.as_ptr() as *mut *const c_void,
            taps_list.len() as _,
            &kCFTypeArrayCallBacks,
        )
        .expect("tap list");

        let dict = CFMutableDictionary::new(
            kCFAllocatorDefault,
            5,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("aggregate dictionary");
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*cf_key(kAudioAggregateDeviceNameKey) as *const _ as *const c_void,
            &*CFString::from_str(agg_name) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*cf_key(kAudioAggregateDeviceUIDKey) as *const _ as *const c_void,
            &*CFString::from_str(agg_uid) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*cf_key(kAudioAggregateDeviceTapListKey) as *const _ as *const c_void,
            &*taps as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*cf_key(kAudioAggregateDeviceTapAutoStartKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*cf_key(kAudioEndPointDeviceIsPrivateKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );
        CFRetained::cast_unchecked::<CFDictionary>(dict)
    }
}

fn default_output_uid() -> Result<objc2::rc::Retained<NSString>> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let uid = device.id().context("query default output UID")?;
    Ok(NSString::from_str(uid.id()))
}

fn cf_key(key: &CStr) -> CFRetained<CFString> {
    CFString::from_str(key.to_str().expect("ASCII Core Audio key"))
}

fn global_address(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn check_os_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("Core Audio OSStatus {status}"))
    }
}

/// Wait until cpal can open the private aggregate as an input device.
pub fn find_aggregate_input(uid: &str) -> Result<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    for _ in 0..15 {
        if let Ok(devices) = host.devices() {
            for device in devices {
                if device.id().ok().is_some_and(|id| id.id() == uid) {
                    if device.supports_input() {
                        return Ok(device);
                    }
                    anyhow::bail!("tap aggregate '{uid}' is visible but has no input stream");
                }
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(anyhow!(
        "tap aggregate '{uid}' did not appear as a cpal device"
    ))
}
