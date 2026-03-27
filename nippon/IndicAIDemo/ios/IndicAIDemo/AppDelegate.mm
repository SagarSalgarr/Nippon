#import "AppDelegate.h"

#import <React/RCTBundleURLProvider.h>

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
  // DIAGNOSTIC: write a file to confirm AppDelegate ran
  NSString *tmp = NSTemporaryDirectory();
  [@"AppDelegate started" writeToFile:[tmp stringByAppendingPathComponent:@"diag_1_appdelegate.txt"]
                           atomically:YES encoding:NSUTF8StringEncoding error:nil];
  NSLog(@"[DIAG] AppDelegate didFinishLaunchingWithOptions — START");

  self.moduleName = @"IndicAIDemo";
  self.initialProps = @{};

  NSURL *bundleURL = [self bundleURL];
  NSLog(@"[DIAG] Bundle URL: %@  exists=%d", bundleURL,
        bundleURL ? [[NSFileManager defaultManager] fileExistsAtPath:bundleURL.path] : NO);
  [@(bundleURL ? [[NSFileManager defaultManager] fileExistsAtPath:bundleURL.path] : NO).description
    writeToFile:[tmp stringByAppendingPathComponent:@"diag_2_bundle_exists.txt"]
     atomically:YES encoding:NSUTF8StringEncoding error:nil];

  BOOL result = [super application:application didFinishLaunchingWithOptions:launchOptions];
  NSLog(@"[DIAG] AppDelegate didFinishLaunchingWithOptions — DONE result=%d", result);
  [@"AppDelegate done" writeToFile:[tmp stringByAppendingPathComponent:@"diag_3_appdelegate_done.txt"]
                        atomically:YES encoding:NSUTF8StringEncoding error:nil];
  return result;
}

- (NSURL *)sourceURLForBridge:(RCTBridge *)bridge
{
  return [self bundleURL];
}

- (NSURL *)bundleURL
{
#if DEBUG
  return [[RCTBundleURLProvider sharedSettings] jsBundleURLForBundleRoot:@"index"];
#else
  return [[NSBundle mainBundle] URLForResource:@"main" withExtension:@"jsbundle"];
#endif
}

@end
