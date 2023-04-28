import subprocess
import json
import random
import signal
import sys
import os
import concurrent.futures

class OnlineJudge(object):
    def __init__(self):
        self.pid_dict = {}
        base_dir = "../../dataset/"
        for contest in os.listdir(base_dir):
            if contest.find("abc") == -1:
                continue
            for p in os.listdir(base_dir + "/" + contest):
                if p.lower() not in ["a", "b", "c", "d"]:
                    continue
                pid = contest + "-" + p
                if not os.path.exists(base_dir + "/" + contest + "/" + p + "/problem_statment"):
                    continue
                self.pid_dict[pid] = {}
                self.pid_dict[pid]["in"] = []
                self.pid_dict[pid]["out"] = []
                for data_point in os.listdir(base_dir + "/" + contest + "/" + p + "/in/"):
                    in_file = base_dir + "/" + contest + "/" + p + "/in/"+ "/" + data_point
                    out_file = base_dir + "/" + contest + "/" + p + "/out/"+ "/" + data_point
                    if not os.path.exists(out_file):
                        continue
                    self.pid_dict[pid]["in"].append(open(in_file).read())
                    self.pid_dict[pid]["out"].append(open(out_file).read())
                self.pid_dict[pid]["problem_statment"] = \
                        open(base_dir + "/" + contest + "/" + p + "/problem_statment").read()

    def check_language(self, code_string):
        if code_string.find("#include") != -1:
            return "c++"
        if code_string.find("raw_input") != -1:
            return "python2.7"
        if code_string.find("print ") != -1:
            return "python2.7"
        return "python3.8"
    
    def run_test_case(self, lan_bin, code_string, one_in, one_out):
        if lan_bin == "c++":
            file_name = "tmp/" + str(random.randint(1, 10000000000))
            f = open(file_name + ".cpp", "w+")
            f.write(code_string)
            f.close()
            p = subprocess.Popen(lan_bin + " -o " + file_name + " " + file_name + ".cpp" + " && " + "./" + file_name,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                shell=True)
        else:
            p = subprocess.Popen([lan_bin, '-c', code_string], 
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid)
        input_string = one_in.encode()
        ce = False
        try:
            output, errors = p.communicate(input_string, timeout=5)
            if errors != b"":
                ce = True
            #print(output, one_out.decode(), errors)
        except:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            output = b"" 
        if ce:
            return -1
        if lan_bin == "c++":
            try:
                os.remove(file_name + ".cpp")
            except:
                pass
            try:
                os.remove(file_name)
            except:
                pass
        score = 0
        if output.decode() == one_out:
            return 1
        return 0

    def run(self, pid, code_string):
        lan_bin = self.check_language(code_string)
        all_tests = 0
        correct_tests = 0
        test_cases = zip(self.pid_dict[pid]["in"], self.pid_dict[pid]["out"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = [executor.submit(self.run_test_case, lan_bin, code_string, one_in, one_out) for (one_in, one_out) in test_cases]
            for result in results:
                correct_tests += result.result()
                all_tests += 1
        return all_tests, correct_tests
    
    def score(self, pid, code_string):
        all_tests, correct_tests = self.run(pid, code_string)
        return int(5.0 * correct_tests / all_tests)

if __name__ == "__main__":
    oj = OnlineJudge()
    print(oj.run("abc260-A", "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  string s;\n  cin >> s;\n  for (int i = 0; i < s.length(); i++) {\n    if (s[i] == s[i + 1] && s[i] == s[i + 2]) {\n      cout << s[i];\n      return 0;\n    }\n  }\n  cout << -1;\n  return 0\n}\n"))
